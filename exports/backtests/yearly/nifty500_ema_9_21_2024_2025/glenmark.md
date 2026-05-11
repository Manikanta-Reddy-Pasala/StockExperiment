# Glenmark Pharmaceuticals Ltd. (GLENMARK)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 2361.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 100 |
| ALERT2 | 99 |
| ALERT2_SKIP | 43 |
| ALERT3 | 263 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 123 |
| PARTIAL | 12 |
| TARGET_HIT | 1 |
| STOP_HIT | 124 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 85
- **Target hits / Stop hits / Partials:** 1 / 124 / 12
- **Avg / median % per leg:** 0.02% / -0.98%
- **Sum % (uncompounded):** 3.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 14 | 25.0% | 1 | 55 | 0 | -0.51% | -28.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 56 | 14 | 25.0% | 1 | 55 | 0 | -0.51% | -28.7% |
| SELL (all) | 81 | 38 | 46.9% | 0 | 69 | 12 | 0.39% | 31.9% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.71% | 1.4% |
| SELL @ 3rd Alert (retest2) | 79 | 36 | 45.6% | 0 | 67 | 12 | 0.39% | 30.5% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.71% | 1.4% |
| retest2 (combined) | 135 | 50 | 37.0% | 1 | 122 | 12 | 0.01% | 1.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 1013.60 | 1016.57 | 1016.90 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1021.25 | 1017.51 | 1017.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 11:15:00 | 1028.30 | 1019.67 | 1018.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 09:15:00 | 1018.65 | 1025.85 | 1022.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 1018.65 | 1025.85 | 1022.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1018.65 | 1025.85 | 1022.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:00:00 | 1018.65 | 1025.85 | 1022.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 1016.15 | 1023.91 | 1021.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:00:00 | 1016.15 | 1023.91 | 1021.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 13:15:00 | 1012.55 | 1019.80 | 1020.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 15:15:00 | 1011.00 | 1016.95 | 1018.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 12:15:00 | 1008.05 | 1006.45 | 1012.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-15 12:45:00 | 1008.10 | 1006.45 | 1012.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 1000.55 | 1004.80 | 1009.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:15:00 | 989.00 | 1003.02 | 1007.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 10:15:00 | 1018.00 | 1006.95 | 1007.50 | SL hit (close>static) qty=1.00 sl=1011.95 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 1017.90 | 1009.14 | 1008.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 15:15:00 | 1020.10 | 1014.00 | 1011.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 11:15:00 | 1029.55 | 1031.25 | 1023.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 12:00:00 | 1029.55 | 1031.25 | 1023.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 1025.50 | 1030.10 | 1023.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:00:00 | 1025.50 | 1030.10 | 1023.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 1028.40 | 1031.24 | 1027.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:00:00 | 1028.40 | 1031.24 | 1027.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 1028.45 | 1030.68 | 1027.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 1028.45 | 1030.68 | 1027.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 1023.15 | 1029.18 | 1027.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:45:00 | 1022.00 | 1029.18 | 1027.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 1030.55 | 1029.45 | 1027.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 1023.75 | 1029.45 | 1027.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1021.20 | 1027.80 | 1026.89 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 09:15:00 | 1012.10 | 1025.88 | 1026.71 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 12:15:00 | 1032.50 | 1027.67 | 1027.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 1115.60 | 1048.31 | 1037.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 1167.35 | 1167.99 | 1143.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-30 09:45:00 | 1167.80 | 1167.99 | 1143.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1151.30 | 1160.24 | 1148.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:30:00 | 1160.65 | 1160.24 | 1148.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1151.40 | 1158.27 | 1149.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 1151.40 | 1158.27 | 1149.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1155.25 | 1157.67 | 1150.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:30:00 | 1151.60 | 1157.67 | 1150.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1170.00 | 1161.69 | 1155.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 13:00:00 | 1172.20 | 1164.25 | 1158.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 1143.70 | 1165.10 | 1161.24 | SL hit (close<static) qty=1.00 sl=1154.35 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1093.70 | 1150.82 | 1155.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1062.10 | 1133.07 | 1146.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 1133.95 | 1131.92 | 1143.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 13:15:00 | 1133.95 | 1131.92 | 1143.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 1133.95 | 1131.92 | 1143.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:45:00 | 1143.85 | 1131.92 | 1143.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1179.90 | 1139.25 | 1143.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 1179.90 | 1139.25 | 1143.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 1194.30 | 1150.26 | 1148.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 1208.90 | 1187.47 | 1174.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 09:15:00 | 1192.45 | 1192.71 | 1179.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 09:45:00 | 1193.05 | 1192.71 | 1179.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 1190.45 | 1193.83 | 1186.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:45:00 | 1190.55 | 1193.83 | 1186.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1208.20 | 1203.29 | 1196.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 10:15:00 | 1214.40 | 1201.10 | 1198.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:30:00 | 1213.50 | 1204.40 | 1200.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 1213.40 | 1204.40 | 1200.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 14:15:00 | 1229.15 | 1240.56 | 1240.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 1229.15 | 1240.56 | 1240.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 15:15:00 | 1228.35 | 1238.11 | 1239.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 1220.95 | 1215.45 | 1220.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 10:15:00 | 1220.95 | 1215.45 | 1220.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1220.95 | 1215.45 | 1220.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:45:00 | 1223.00 | 1215.45 | 1220.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 1232.75 | 1218.91 | 1221.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:30:00 | 1235.90 | 1218.91 | 1221.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 1226.60 | 1220.45 | 1222.27 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 1226.45 | 1223.42 | 1223.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 1234.05 | 1226.51 | 1224.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 10:15:00 | 1225.65 | 1226.34 | 1224.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 10:15:00 | 1225.65 | 1226.34 | 1224.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 1225.65 | 1226.34 | 1224.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 1226.65 | 1226.34 | 1224.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1223.95 | 1225.86 | 1224.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:45:00 | 1224.35 | 1225.86 | 1224.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1226.10 | 1225.91 | 1224.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:45:00 | 1220.35 | 1225.91 | 1224.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 1216.30 | 1223.99 | 1224.18 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 1230.00 | 1224.84 | 1224.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 1245.10 | 1228.89 | 1226.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 11:15:00 | 1226.45 | 1230.77 | 1227.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 11:15:00 | 1226.45 | 1230.77 | 1227.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 1226.45 | 1230.77 | 1227.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 1226.45 | 1230.77 | 1227.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 1224.35 | 1229.49 | 1227.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:00:00 | 1224.35 | 1229.49 | 1227.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 1231.30 | 1229.85 | 1227.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:30:00 | 1225.25 | 1229.85 | 1227.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1233.00 | 1230.52 | 1228.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 1238.05 | 1230.52 | 1228.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-08 09:15:00 | 1361.86 | 1335.26 | 1312.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 11:15:00 | 1419.65 | 1423.78 | 1424.03 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 15:15:00 | 1436.65 | 1425.86 | 1424.80 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 13:15:00 | 1423.00 | 1431.51 | 1431.76 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 14:15:00 | 1442.85 | 1433.78 | 1432.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 1458.70 | 1441.15 | 1437.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 1457.20 | 1459.58 | 1450.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 10:00:00 | 1457.20 | 1459.58 | 1450.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1451.10 | 1457.88 | 1450.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 1451.10 | 1457.88 | 1450.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 1438.30 | 1453.96 | 1449.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 1438.30 | 1453.96 | 1449.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 1451.00 | 1453.37 | 1449.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 1441.20 | 1453.37 | 1449.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 1442.05 | 1451.11 | 1449.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:30:00 | 1440.50 | 1451.11 | 1449.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1439.00 | 1448.69 | 1448.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:45:00 | 1428.05 | 1448.69 | 1448.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 1439.00 | 1446.75 | 1447.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 1422.60 | 1439.57 | 1443.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1448.15 | 1430.56 | 1435.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1448.15 | 1430.56 | 1435.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1448.15 | 1430.56 | 1435.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 1449.20 | 1430.56 | 1435.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1454.00 | 1435.24 | 1437.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 1454.00 | 1435.24 | 1437.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 1461.00 | 1440.40 | 1439.44 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 1423.75 | 1438.03 | 1438.79 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 09:15:00 | 1459.80 | 1441.24 | 1440.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 1501.20 | 1475.30 | 1465.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 10:15:00 | 1494.10 | 1499.30 | 1486.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 10:30:00 | 1498.55 | 1499.30 | 1486.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 1487.00 | 1496.84 | 1486.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 1487.00 | 1496.84 | 1486.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1467.90 | 1491.05 | 1484.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 1473.75 | 1491.05 | 1484.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1485.25 | 1489.89 | 1484.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 1485.25 | 1489.89 | 1484.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1479.15 | 1487.74 | 1484.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:45:00 | 1477.60 | 1487.74 | 1484.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1483.90 | 1486.97 | 1484.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 1506.85 | 1486.97 | 1484.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1492.95 | 1488.17 | 1484.99 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 12:15:00 | 1474.65 | 1482.63 | 1483.03 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 13:15:00 | 1486.20 | 1483.35 | 1483.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 1512.30 | 1491.04 | 1487.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 1662.40 | 1664.33 | 1638.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 10:00:00 | 1662.40 | 1664.33 | 1638.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1710.90 | 1695.04 | 1682.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:15:00 | 1714.85 | 1695.04 | 1682.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 15:15:00 | 1686.05 | 1695.14 | 1696.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 1686.05 | 1695.14 | 1696.21 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1715.25 | 1699.16 | 1697.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 10:15:00 | 1725.35 | 1704.40 | 1700.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 09:15:00 | 1719.25 | 1723.45 | 1713.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 10:00:00 | 1719.25 | 1723.45 | 1713.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1708.75 | 1720.51 | 1713.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 1708.75 | 1720.51 | 1713.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 1705.70 | 1717.55 | 1712.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:45:00 | 1703.70 | 1717.55 | 1712.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 13:15:00 | 1692.40 | 1707.97 | 1708.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 14:15:00 | 1688.00 | 1703.97 | 1707.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 1713.25 | 1703.75 | 1706.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 1713.25 | 1703.75 | 1706.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1713.25 | 1703.75 | 1706.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:00:00 | 1713.25 | 1703.75 | 1706.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 1707.55 | 1704.51 | 1706.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 11:30:00 | 1704.05 | 1704.22 | 1706.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 11:15:00 | 1713.90 | 1701.04 | 1699.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 1713.90 | 1701.04 | 1699.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 1729.60 | 1710.25 | 1706.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 1724.70 | 1726.33 | 1720.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 1724.70 | 1726.33 | 1720.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1726.25 | 1726.31 | 1721.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 1726.25 | 1726.31 | 1721.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1732.10 | 1739.11 | 1732.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:00:00 | 1732.10 | 1739.11 | 1732.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 1744.90 | 1740.27 | 1733.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:45:00 | 1749.90 | 1740.85 | 1734.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 13:30:00 | 1750.30 | 1743.94 | 1737.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 1751.25 | 1746.22 | 1739.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 10:30:00 | 1750.70 | 1746.59 | 1740.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 1741.00 | 1745.22 | 1741.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:00:00 | 1741.00 | 1745.22 | 1741.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 1742.30 | 1744.63 | 1741.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:15:00 | 1740.40 | 1744.63 | 1741.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 1742.90 | 1744.29 | 1741.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:45:00 | 1737.00 | 1744.29 | 1741.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 1744.10 | 1744.25 | 1741.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 1741.45 | 1744.25 | 1741.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1728.05 | 1741.01 | 1740.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 1728.05 | 1741.01 | 1740.41 | SL hit (close<static) qty=1.00 sl=1731.50 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 1725.25 | 1737.86 | 1739.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 14:15:00 | 1712.25 | 1726.03 | 1732.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 1649.10 | 1646.42 | 1670.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 1649.10 | 1646.42 | 1670.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1649.80 | 1648.83 | 1667.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:15:00 | 1632.00 | 1648.83 | 1667.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:00:00 | 1637.70 | 1645.38 | 1659.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:30:00 | 1630.90 | 1643.35 | 1657.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 1744.70 | 1659.08 | 1662.01 | SL hit (close>static) qty=1.00 sl=1674.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 1723.00 | 1671.87 | 1667.55 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 1688.10 | 1689.80 | 1690.00 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 15:15:00 | 1692.00 | 1690.24 | 1690.18 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 1677.00 | 1687.59 | 1688.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 1669.80 | 1682.27 | 1686.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 1688.75 | 1680.46 | 1683.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 1688.75 | 1680.46 | 1683.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1688.75 | 1680.46 | 1683.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 1691.40 | 1680.46 | 1683.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1682.00 | 1680.77 | 1683.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 1688.65 | 1680.77 | 1683.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 1687.95 | 1682.21 | 1683.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:30:00 | 1687.80 | 1682.21 | 1683.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 1685.10 | 1682.79 | 1684.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:15:00 | 1696.50 | 1682.79 | 1684.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 1693.45 | 1684.92 | 1684.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:30:00 | 1693.95 | 1684.92 | 1684.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 1686.60 | 1685.25 | 1685.07 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 1680.05 | 1684.65 | 1684.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 10:15:00 | 1670.20 | 1681.76 | 1683.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 11:15:00 | 1685.80 | 1682.57 | 1683.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 11:15:00 | 1685.80 | 1682.57 | 1683.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 1685.80 | 1682.57 | 1683.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:45:00 | 1688.60 | 1682.57 | 1683.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 1690.15 | 1684.09 | 1684.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:45:00 | 1688.20 | 1684.09 | 1684.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 1675.65 | 1682.40 | 1683.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 15:00:00 | 1671.25 | 1680.17 | 1682.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 1669.35 | 1680.93 | 1682.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:45:00 | 1668.80 | 1668.03 | 1673.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 11:15:00 | 1693.00 | 1670.47 | 1669.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 11:15:00 | 1693.00 | 1670.47 | 1669.13 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 12:15:00 | 1660.10 | 1668.44 | 1669.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 13:15:00 | 1659.55 | 1666.67 | 1668.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 14:15:00 | 1672.05 | 1667.74 | 1668.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 14:15:00 | 1672.05 | 1667.74 | 1668.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1672.05 | 1667.74 | 1668.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 15:00:00 | 1672.05 | 1667.74 | 1668.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 15:15:00 | 1677.00 | 1669.59 | 1669.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 09:15:00 | 1709.90 | 1677.66 | 1673.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 1767.00 | 1774.29 | 1749.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 1767.00 | 1774.29 | 1749.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 1765.00 | 1770.03 | 1751.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:30:00 | 1750.20 | 1770.03 | 1751.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 1804.15 | 1809.57 | 1800.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:30:00 | 1797.15 | 1809.57 | 1800.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1775.25 | 1801.83 | 1798.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 1772.00 | 1801.83 | 1798.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 1767.00 | 1794.86 | 1795.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 1753.60 | 1786.61 | 1791.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 1744.45 | 1738.51 | 1754.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 14:00:00 | 1744.45 | 1738.51 | 1754.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1724.30 | 1735.96 | 1749.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 1717.20 | 1728.89 | 1743.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 1710.25 | 1721.00 | 1733.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 13:15:00 | 1714.60 | 1690.64 | 1687.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 1714.60 | 1690.64 | 1687.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 15:15:00 | 1722.90 | 1700.15 | 1692.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 1664.45 | 1693.01 | 1689.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 09:15:00 | 1664.45 | 1693.01 | 1689.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 1664.45 | 1693.01 | 1689.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 1664.45 | 1693.01 | 1689.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 1650.65 | 1684.54 | 1686.36 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 14:15:00 | 1695.45 | 1682.11 | 1680.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 18:15:00 | 1699.50 | 1690.40 | 1685.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1666.10 | 1685.54 | 1683.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1666.10 | 1685.54 | 1683.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1666.10 | 1685.54 | 1683.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:30:00 | 1680.55 | 1685.54 | 1683.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 1654.60 | 1679.35 | 1680.74 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 13:15:00 | 1695.75 | 1681.02 | 1680.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 15:15:00 | 1707.45 | 1688.46 | 1684.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 11:15:00 | 1684.20 | 1691.03 | 1686.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 11:15:00 | 1684.20 | 1691.03 | 1686.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 1684.20 | 1691.03 | 1686.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:00:00 | 1684.20 | 1691.03 | 1686.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 1688.00 | 1690.43 | 1687.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:15:00 | 1699.00 | 1690.43 | 1687.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 11:15:00 | 1672.15 | 1712.09 | 1713.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 1672.15 | 1712.09 | 1713.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 13:15:00 | 1652.20 | 1693.23 | 1703.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 13:15:00 | 1674.35 | 1674.21 | 1686.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 14:00:00 | 1674.35 | 1674.21 | 1686.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 1681.50 | 1670.70 | 1680.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:30:00 | 1680.05 | 1670.70 | 1680.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1658.40 | 1668.24 | 1678.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 1675.15 | 1668.24 | 1678.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1530.15 | 1540.59 | 1561.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:00:00 | 1507.05 | 1531.25 | 1550.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 11:45:00 | 1506.05 | 1509.37 | 1529.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:15:00 | 1498.75 | 1510.05 | 1526.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 13:15:00 | 1495.95 | 1490.60 | 1489.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 13:15:00 | 1495.95 | 1490.60 | 1489.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 1505.00 | 1493.46 | 1491.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 13:15:00 | 1516.20 | 1516.57 | 1509.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 13:45:00 | 1514.40 | 1516.57 | 1509.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1497.95 | 1514.25 | 1510.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:00:00 | 1497.95 | 1514.25 | 1510.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1502.65 | 1511.93 | 1510.18 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 1498.20 | 1507.20 | 1508.21 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 1515.90 | 1507.66 | 1507.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 14:15:00 | 1528.85 | 1516.06 | 1511.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 09:15:00 | 1552.95 | 1554.67 | 1544.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 09:45:00 | 1550.80 | 1554.67 | 1544.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1555.80 | 1555.01 | 1546.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 1550.00 | 1555.01 | 1546.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1546.00 | 1553.21 | 1546.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:00:00 | 1546.00 | 1553.21 | 1546.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 1545.45 | 1551.66 | 1546.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:15:00 | 1543.25 | 1551.66 | 1546.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1548.35 | 1551.00 | 1546.57 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 1523.50 | 1543.80 | 1544.26 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 1547.55 | 1544.11 | 1544.04 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 1530.60 | 1541.87 | 1543.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 11:15:00 | 1526.90 | 1537.51 | 1540.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 15:15:00 | 1520.05 | 1518.32 | 1525.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 09:15:00 | 1566.40 | 1518.32 | 1525.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1549.00 | 1524.46 | 1528.04 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 1548.25 | 1533.27 | 1531.68 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 1525.20 | 1533.75 | 1534.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1465.95 | 1518.63 | 1526.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 1515.00 | 1508.05 | 1518.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 1515.00 | 1508.05 | 1518.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 1517.30 | 1509.90 | 1518.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 1540.05 | 1509.90 | 1518.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1518.75 | 1511.67 | 1518.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:45:00 | 1526.00 | 1511.67 | 1518.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 1514.05 | 1512.15 | 1518.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 1530.35 | 1512.15 | 1518.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1542.60 | 1518.24 | 1520.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 1548.15 | 1518.24 | 1520.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1532.15 | 1521.02 | 1521.42 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 1547.25 | 1526.27 | 1523.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 12:15:00 | 1561.40 | 1533.29 | 1527.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 09:15:00 | 1541.50 | 1542.19 | 1534.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 1541.50 | 1542.19 | 1534.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1541.50 | 1542.19 | 1534.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:30:00 | 1535.95 | 1542.19 | 1534.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1528.15 | 1539.39 | 1533.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:45:00 | 1524.45 | 1539.39 | 1533.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 1531.30 | 1537.77 | 1533.39 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 1511.65 | 1530.72 | 1530.84 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 13:15:00 | 1530.00 | 1529.43 | 1529.42 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 14:15:00 | 1524.80 | 1528.51 | 1529.00 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 10:15:00 | 1537.65 | 1529.05 | 1529.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 11:15:00 | 1541.40 | 1531.52 | 1530.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 1543.05 | 1543.88 | 1539.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 1543.05 | 1543.88 | 1539.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1543.05 | 1543.88 | 1539.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 1543.05 | 1543.88 | 1539.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1545.00 | 1544.10 | 1540.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:45:00 | 1538.65 | 1543.50 | 1540.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1553.00 | 1545.40 | 1541.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 1556.15 | 1545.40 | 1541.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 10:45:00 | 1556.50 | 1549.31 | 1545.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 11:15:00 | 1557.90 | 1549.31 | 1545.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 14:15:00 | 1534.75 | 1545.26 | 1544.96 | SL hit (close<static) qty=1.00 sl=1539.85 alert=retest2 |

### Cycle 57 — SELL (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 15:15:00 | 1533.95 | 1542.99 | 1543.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 1518.25 | 1538.05 | 1541.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 14:15:00 | 1545.35 | 1531.77 | 1536.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 14:15:00 | 1545.35 | 1531.77 | 1536.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1545.35 | 1531.77 | 1536.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1545.35 | 1531.77 | 1536.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1544.00 | 1534.21 | 1536.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1573.00 | 1534.21 | 1536.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1574.40 | 1542.25 | 1540.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 10:15:00 | 1588.00 | 1551.40 | 1544.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 1581.50 | 1583.76 | 1571.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 13:45:00 | 1579.25 | 1583.76 | 1571.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1599.25 | 1609.19 | 1601.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:45:00 | 1600.30 | 1609.19 | 1601.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 1609.10 | 1609.17 | 1602.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:45:00 | 1624.45 | 1611.95 | 1604.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 11:15:00 | 1616.95 | 1629.69 | 1624.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 11:45:00 | 1616.00 | 1629.55 | 1624.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 12:15:00 | 1606.20 | 1630.91 | 1631.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 1606.20 | 1630.91 | 1631.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 13:15:00 | 1598.45 | 1617.16 | 1623.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 12:15:00 | 1512.65 | 1511.32 | 1535.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 12:45:00 | 1509.30 | 1511.32 | 1535.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1473.65 | 1505.52 | 1525.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 1440.15 | 1468.13 | 1484.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:45:00 | 1446.60 | 1463.58 | 1479.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 1450.40 | 1463.35 | 1473.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 12:15:00 | 1500.60 | 1476.55 | 1476.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 1500.60 | 1476.55 | 1476.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 1507.55 | 1482.75 | 1479.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 1498.60 | 1502.61 | 1494.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 15:00:00 | 1498.60 | 1502.61 | 1494.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1489.90 | 1499.98 | 1494.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 1492.00 | 1499.98 | 1494.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 1491.35 | 1498.25 | 1494.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:15:00 | 1486.20 | 1498.25 | 1494.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 1478.90 | 1494.38 | 1492.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 1478.90 | 1494.38 | 1492.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 1486.60 | 1492.83 | 1492.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:30:00 | 1483.95 | 1492.83 | 1492.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1515.00 | 1499.60 | 1495.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 10:15:00 | 1517.05 | 1499.60 | 1495.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 12:00:00 | 1519.55 | 1506.19 | 1499.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 09:15:00 | 1465.60 | 1500.22 | 1499.82 | SL hit (close<static) qty=1.00 sl=1485.55 alert=retest2 |

### Cycle 61 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 1472.90 | 1494.76 | 1497.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 12:15:00 | 1456.05 | 1483.36 | 1491.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 13:15:00 | 1415.30 | 1402.94 | 1424.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:45:00 | 1412.55 | 1402.94 | 1424.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1433.55 | 1409.99 | 1422.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1433.55 | 1409.99 | 1422.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1440.60 | 1416.11 | 1424.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 1440.60 | 1416.11 | 1424.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 1452.00 | 1432.54 | 1430.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 1453.80 | 1436.79 | 1432.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 1443.40 | 1446.48 | 1439.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 11:15:00 | 1443.40 | 1446.48 | 1439.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 1443.40 | 1446.48 | 1439.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:30:00 | 1440.65 | 1446.48 | 1439.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 1448.10 | 1454.02 | 1447.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 13:00:00 | 1448.10 | 1454.02 | 1447.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 1446.40 | 1452.50 | 1447.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:00:00 | 1446.40 | 1452.50 | 1447.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1453.70 | 1452.74 | 1448.06 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 11:15:00 | 1429.15 | 1443.22 | 1444.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1412.95 | 1433.36 | 1438.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 1416.00 | 1415.84 | 1426.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 15:00:00 | 1416.00 | 1415.84 | 1426.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1425.35 | 1417.57 | 1425.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 1438.40 | 1417.57 | 1425.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1431.00 | 1420.26 | 1426.09 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 1442.95 | 1429.63 | 1429.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 1471.15 | 1443.47 | 1436.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 12:15:00 | 1526.20 | 1527.37 | 1512.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-10 12:30:00 | 1526.10 | 1527.37 | 1512.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 1503.80 | 1519.22 | 1511.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 1477.70 | 1519.22 | 1511.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 1454.50 | 1506.27 | 1506.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 1451.30 | 1495.28 | 1501.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1428.65 | 1422.70 | 1445.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:45:00 | 1434.55 | 1422.70 | 1445.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1354.55 | 1402.37 | 1424.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 1348.00 | 1402.37 | 1424.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1280.60 | 1351.19 | 1381.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 1372.30 | 1351.19 | 1381.32 | SL hit (close>static) qty=0.50 sl=1351.19 alert=retest2 |

### Cycle 66 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 1314.60 | 1303.58 | 1302.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 1334.00 | 1315.32 | 1308.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 10:15:00 | 1389.10 | 1396.21 | 1385.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 1389.10 | 1396.21 | 1385.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1419.15 | 1402.78 | 1393.56 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 1387.25 | 1394.36 | 1394.58 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 12:15:00 | 1398.85 | 1395.26 | 1394.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 13:15:00 | 1405.55 | 1397.31 | 1395.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 10:15:00 | 1404.75 | 1406.66 | 1401.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 10:15:00 | 1404.75 | 1406.66 | 1401.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1404.75 | 1406.66 | 1401.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:00:00 | 1404.75 | 1406.66 | 1401.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 1404.00 | 1406.13 | 1401.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:30:00 | 1400.90 | 1406.13 | 1401.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 1402.05 | 1405.31 | 1401.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:30:00 | 1402.40 | 1405.31 | 1401.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 1407.30 | 1405.71 | 1402.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 1408.50 | 1405.71 | 1402.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 1427.40 | 1405.65 | 1402.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 14:15:00 | 1479.25 | 1486.00 | 1486.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 1479.25 | 1486.00 | 1486.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 1476.00 | 1483.52 | 1485.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 1478.40 | 1471.54 | 1476.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 1478.40 | 1471.54 | 1476.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1478.40 | 1471.54 | 1476.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 1478.40 | 1471.54 | 1476.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 1495.00 | 1476.23 | 1478.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:00:00 | 1495.00 | 1476.23 | 1478.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 1493.50 | 1479.68 | 1479.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 13:15:00 | 1501.35 | 1484.02 | 1481.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 1509.75 | 1523.42 | 1511.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 10:15:00 | 1509.75 | 1523.42 | 1511.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1509.75 | 1523.42 | 1511.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 1509.75 | 1523.42 | 1511.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 1518.60 | 1522.45 | 1512.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 10:45:00 | 1525.20 | 1514.27 | 1511.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:00:00 | 1523.75 | 1516.17 | 1512.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:00:00 | 1522.00 | 1518.39 | 1514.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 1563.30 | 1518.73 | 1515.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1475.00 | 1523.09 | 1521.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 1475.00 | 1523.09 | 1521.94 | SL hit (close<static) qty=1.00 sl=1507.75 alert=retest2 |

### Cycle 71 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1499.95 | 1518.47 | 1519.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1445.95 | 1493.05 | 1505.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 1382.80 | 1381.52 | 1399.40 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-15 11:45:00 | 1369.80 | 1379.78 | 1395.47 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 09:45:00 | 1370.00 | 1375.64 | 1387.55 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1350.70 | 1359.43 | 1372.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 10:30:00 | 1348.90 | 1357.54 | 1370.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 11:45:00 | 1345.40 | 1354.91 | 1367.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 11:15:00 | 1360.20 | 1351.44 | 1358.81 | SL hit (close>ema400) qty=1.00 sl=1358.81 alert=retest1 |

### Cycle 72 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 1386.70 | 1367.15 | 1364.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 10:15:00 | 1400.10 | 1373.74 | 1367.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 1384.20 | 1412.15 | 1404.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 1384.20 | 1412.15 | 1404.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1384.20 | 1412.15 | 1404.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1384.20 | 1412.15 | 1404.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1361.90 | 1402.10 | 1401.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 1361.90 | 1402.10 | 1401.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1380.40 | 1397.76 | 1399.13 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 1405.20 | 1392.83 | 1392.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 1406.40 | 1395.55 | 1393.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 09:15:00 | 1391.90 | 1397.91 | 1395.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 1391.90 | 1397.91 | 1395.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1391.90 | 1397.91 | 1395.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:45:00 | 1390.00 | 1397.91 | 1395.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 1400.00 | 1398.33 | 1395.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:30:00 | 1387.20 | 1398.33 | 1395.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 1395.30 | 1397.73 | 1395.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 1395.30 | 1397.73 | 1395.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 1396.00 | 1397.38 | 1395.57 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 14:15:00 | 1384.80 | 1393.92 | 1394.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 1376.60 | 1389.70 | 1392.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1392.70 | 1385.74 | 1388.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1392.70 | 1385.74 | 1388.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1392.70 | 1385.74 | 1388.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 1392.70 | 1385.74 | 1388.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 1388.20 | 1386.23 | 1388.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 1391.10 | 1386.23 | 1388.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 1390.40 | 1387.07 | 1388.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:45:00 | 1394.80 | 1387.07 | 1388.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 1379.50 | 1385.55 | 1388.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 13:45:00 | 1376.00 | 1383.44 | 1386.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:15:00 | 1377.00 | 1377.23 | 1381.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 1401.50 | 1382.08 | 1383.31 | SL hit (close>static) qty=1.00 sl=1391.30 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 1406.90 | 1387.05 | 1385.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 1410.00 | 1393.73 | 1388.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 13:15:00 | 1405.20 | 1405.42 | 1396.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 14:00:00 | 1405.20 | 1405.42 | 1396.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 1392.40 | 1402.13 | 1397.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:15:00 | 1425.00 | 1404.28 | 1399.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 10:15:00 | 1398.40 | 1414.25 | 1414.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 1398.40 | 1414.25 | 1414.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-12 09:15:00 | 1372.60 | 1398.51 | 1406.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 1423.30 | 1393.39 | 1398.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 1423.30 | 1393.39 | 1398.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 1423.30 | 1393.39 | 1398.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:30:00 | 1419.30 | 1393.39 | 1398.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1414.70 | 1397.66 | 1399.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 1424.50 | 1397.66 | 1399.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 11:15:00 | 1420.10 | 1402.14 | 1401.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 1433.10 | 1415.01 | 1408.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 11:15:00 | 1439.30 | 1440.07 | 1428.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:00:00 | 1439.30 | 1440.07 | 1428.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1436.50 | 1440.37 | 1434.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 1432.50 | 1440.37 | 1434.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1450.80 | 1442.46 | 1435.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 12:15:00 | 1454.00 | 1442.46 | 1435.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:45:00 | 1464.80 | 1450.33 | 1442.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:45:00 | 1455.10 | 1450.52 | 1443.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:15:00 | 1457.90 | 1450.52 | 1443.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 1441.60 | 1448.52 | 1444.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 1441.60 | 1448.52 | 1444.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1432.10 | 1445.24 | 1443.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-19 14:15:00 | 1432.10 | 1445.24 | 1443.07 | SL hit (close<static) qty=1.00 sl=1434.50 alert=retest2 |

### Cycle 79 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 1442.00 | 1443.52 | 1443.58 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 1453.00 | 1445.42 | 1444.43 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1433.00 | 1443.59 | 1443.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 1425.60 | 1437.12 | 1440.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 1432.50 | 1432.05 | 1436.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 14:45:00 | 1434.10 | 1432.05 | 1436.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1427.50 | 1430.89 | 1434.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:45:00 | 1413.20 | 1427.49 | 1433.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:30:00 | 1413.20 | 1422.48 | 1429.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:45:00 | 1413.00 | 1418.38 | 1425.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 1410.80 | 1394.13 | 1393.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 1410.80 | 1394.13 | 1393.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1431.20 | 1401.55 | 1396.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 1503.20 | 1504.32 | 1481.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:30:00 | 1504.70 | 1504.32 | 1481.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1621.10 | 1615.57 | 1600.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 12:00:00 | 1626.40 | 1618.45 | 1604.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 1657.90 | 1662.86 | 1662.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1657.90 | 1662.86 | 1662.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1645.60 | 1657.95 | 1660.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1643.40 | 1642.02 | 1648.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:45:00 | 1640.10 | 1642.02 | 1648.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1636.60 | 1637.89 | 1644.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 1644.10 | 1637.89 | 1644.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1661.00 | 1641.96 | 1645.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:15:00 | 1674.50 | 1641.96 | 1645.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 1688.00 | 1651.17 | 1648.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1704.00 | 1661.73 | 1653.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1685.10 | 1687.41 | 1676.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1685.10 | 1687.41 | 1676.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1714.00 | 1692.50 | 1680.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:00:00 | 1739.00 | 1712.55 | 1697.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 1729.70 | 1715.76 | 1700.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 1730.00 | 1721.74 | 1706.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 1732.10 | 1725.86 | 1712.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 1717.80 | 1723.61 | 1714.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:00:00 | 1717.80 | 1723.61 | 1714.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 1715.10 | 1721.91 | 1714.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 1715.10 | 1721.91 | 1714.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1716.60 | 1720.85 | 1714.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:15:00 | 1720.00 | 1718.86 | 1714.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 1806.60 | 1822.68 | 1824.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 12:15:00 | 1806.60 | 1822.68 | 1824.17 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 1859.40 | 1828.46 | 1826.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 10:15:00 | 1875.20 | 1837.81 | 1830.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 2178.00 | 2184.42 | 2107.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 10:00:00 | 2178.00 | 2184.42 | 2107.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 2195.50 | 2213.40 | 2188.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:45:00 | 2198.40 | 2213.40 | 2188.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 2200.50 | 2208.17 | 2190.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:30:00 | 2185.20 | 2208.17 | 2190.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 2205.50 | 2217.36 | 2205.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 2204.70 | 2217.36 | 2205.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 2199.00 | 2213.68 | 2204.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 2205.00 | 2213.68 | 2204.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 2199.40 | 2210.83 | 2204.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 2199.40 | 2210.83 | 2204.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 2226.40 | 2213.94 | 2206.40 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 2185.40 | 2205.83 | 2206.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 2165.70 | 2195.19 | 2201.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 12:15:00 | 2152.10 | 2143.27 | 2155.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:15:00 | 2153.30 | 2143.27 | 2155.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 2153.30 | 2145.27 | 2155.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 2150.00 | 2145.27 | 2155.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 2144.20 | 2145.06 | 2154.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:45:00 | 2140.00 | 2147.58 | 2150.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 2154.90 | 2149.05 | 2151.06 | SL hit (close>static) qty=1.00 sl=2154.70 alert=retest2 |

### Cycle 88 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 2172.20 | 2153.68 | 2152.98 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 2148.90 | 2152.70 | 2152.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 2134.40 | 2145.65 | 2149.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 2147.30 | 2144.45 | 2148.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 2147.30 | 2144.45 | 2148.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2147.30 | 2144.45 | 2148.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:45:00 | 2146.30 | 2144.45 | 2148.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2142.50 | 2144.06 | 2147.50 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 2165.70 | 2150.71 | 2150.07 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 2143.50 | 2150.75 | 2151.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 2116.90 | 2140.77 | 2145.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 2088.10 | 2083.09 | 2101.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:45:00 | 2093.20 | 2083.09 | 2101.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 2033.80 | 2029.31 | 2045.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 2037.00 | 2029.31 | 2045.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 2045.00 | 2032.81 | 2042.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 2045.00 | 2032.81 | 2042.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2064.30 | 2039.11 | 2044.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2064.30 | 2039.11 | 2044.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2060.10 | 2043.31 | 2045.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 2042.60 | 2043.31 | 2045.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 2049.60 | 2038.61 | 2038.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 2049.60 | 2038.61 | 2038.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 2068.90 | 2051.39 | 2045.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 15:15:00 | 2053.40 | 2054.94 | 2048.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:15:00 | 2042.70 | 2054.94 | 2048.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 2059.30 | 2055.81 | 2049.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:30:00 | 2044.10 | 2055.81 | 2049.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2059.60 | 2056.57 | 2050.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 2051.00 | 2056.57 | 2050.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 2053.00 | 2055.85 | 2051.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 2053.00 | 2055.85 | 2051.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 2041.80 | 2053.04 | 2050.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:45:00 | 2044.40 | 2053.04 | 2050.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 2024.40 | 2047.32 | 2047.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 09:15:00 | 2022.20 | 2035.38 | 2039.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1947.90 | 1934.29 | 1951.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:45:00 | 1945.50 | 1934.29 | 1951.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1957.00 | 1938.83 | 1952.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 1958.30 | 1938.83 | 1952.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1948.90 | 1940.84 | 1951.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 1940.60 | 1940.84 | 1951.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 14:30:00 | 1940.70 | 1942.30 | 1950.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:45:00 | 1941.80 | 1944.38 | 1949.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:00:00 | 1936.40 | 1942.79 | 1948.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1953.00 | 1942.59 | 1945.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 1953.00 | 1942.59 | 1945.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1956.40 | 1945.35 | 1946.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1956.40 | 1945.35 | 1946.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-25 12:15:00 | 1960.90 | 1949.62 | 1948.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 1960.90 | 1949.62 | 1948.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 12:15:00 | 1969.80 | 1956.83 | 1953.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 1951.30 | 1958.04 | 1954.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 1951.30 | 1958.04 | 1954.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1951.30 | 1958.04 | 1954.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 1951.30 | 1958.04 | 1954.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1950.00 | 1956.43 | 1953.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 1936.90 | 1956.43 | 1953.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 1950.00 | 1952.34 | 1952.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 1934.00 | 1946.35 | 1949.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1942.00 | 1941.57 | 1946.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1942.00 | 1941.57 | 1946.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1926.60 | 1930.18 | 1937.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1933.90 | 1930.18 | 1937.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1930.90 | 1925.74 | 1930.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:45:00 | 1914.50 | 1924.94 | 1928.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 1965.90 | 1932.34 | 1931.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1965.90 | 1932.34 | 1931.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 1991.50 | 1944.17 | 1936.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1989.40 | 1990.72 | 1975.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 1989.40 | 1990.72 | 1975.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 2106.50 | 2120.61 | 2100.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 2090.00 | 2120.61 | 2100.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 2112.50 | 2118.99 | 2101.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 2107.80 | 2118.99 | 2101.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 2127.00 | 2136.24 | 2124.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 2108.80 | 2136.24 | 2124.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 2105.60 | 2130.11 | 2123.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 2097.00 | 2130.11 | 2123.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 2107.50 | 2125.59 | 2121.65 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 2096.60 | 2117.65 | 2118.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 09:15:00 | 2076.30 | 2101.71 | 2110.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 2074.10 | 2058.21 | 2072.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 2074.10 | 2058.21 | 2072.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 2074.10 | 2058.21 | 2072.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 2074.10 | 2058.21 | 2072.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 2069.70 | 2060.50 | 2072.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 2078.50 | 2060.50 | 2072.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 2071.90 | 2062.78 | 2072.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:30:00 | 2070.40 | 2062.78 | 2072.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 2078.20 | 2065.87 | 2072.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 2074.20 | 2065.87 | 2072.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2109.90 | 2074.67 | 2076.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 2109.90 | 2074.67 | 2076.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 2108.30 | 2081.40 | 2079.19 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 2071.10 | 2083.96 | 2084.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 11:15:00 | 2038.80 | 2074.93 | 2080.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 2035.90 | 2035.82 | 2053.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:00:00 | 2035.90 | 2035.82 | 2053.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 2045.10 | 2027.71 | 2037.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 2045.10 | 2027.71 | 2037.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 2025.70 | 2027.31 | 2036.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 2017.30 | 2025.41 | 2035.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1983.00 | 2032.82 | 2035.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 1983.90 | 1965.63 | 1964.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 1983.90 | 1965.63 | 1964.81 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1964.00 | 1965.57 | 1965.60 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 1971.80 | 1966.54 | 1965.89 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 1950.40 | 1963.31 | 1964.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1932.30 | 1955.63 | 1960.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1949.90 | 1941.66 | 1949.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1949.90 | 1941.66 | 1949.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1949.90 | 1941.66 | 1949.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:45:00 | 1934.70 | 1940.39 | 1946.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 1935.10 | 1939.33 | 1945.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 1929.40 | 1938.40 | 1943.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 1964.00 | 1943.38 | 1945.10 | SL hit (close>static) qty=1.00 sl=1961.80 alert=retest2 |

### Cycle 104 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 1839.00 | 1828.57 | 1827.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 1868.00 | 1842.08 | 1834.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 15:15:00 | 1889.10 | 1892.77 | 1882.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 1867.10 | 1892.77 | 1882.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1872.00 | 1888.61 | 1881.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:15:00 | 1863.50 | 1888.61 | 1881.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1857.90 | 1882.47 | 1879.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 1857.90 | 1882.47 | 1879.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 1867.50 | 1876.68 | 1877.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 1859.30 | 1873.21 | 1875.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 1826.60 | 1825.30 | 1842.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:00:00 | 1826.60 | 1825.30 | 1842.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1837.60 | 1821.16 | 1830.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 1844.00 | 1821.16 | 1830.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1827.50 | 1822.43 | 1830.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1830.80 | 1822.43 | 1830.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1835.20 | 1824.98 | 1830.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 1848.50 | 1824.98 | 1830.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 1834.80 | 1826.94 | 1831.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:30:00 | 1832.90 | 1828.28 | 1831.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:00:00 | 1829.60 | 1828.54 | 1831.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1830.80 | 1830.45 | 1831.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:15:00 | 1819.60 | 1831.44 | 1832.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1822.20 | 1829.59 | 1831.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:00:00 | 1817.90 | 1825.78 | 1828.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:00:00 | 1816.70 | 1823.97 | 1827.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1838.10 | 1825.20 | 1827.65 | SL hit (close>static) qty=1.00 sl=1836.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1851.70 | 1830.50 | 1829.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 1875.40 | 1848.61 | 1839.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 1879.90 | 1881.18 | 1862.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 1879.90 | 1881.18 | 1862.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1892.00 | 1884.58 | 1867.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 1910.00 | 1889.66 | 1871.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 1923.20 | 1892.11 | 1873.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:00:00 | 1917.00 | 1902.18 | 1886.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 15:15:00 | 1867.30 | 1882.40 | 1882.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 1867.30 | 1882.40 | 1882.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1836.90 | 1873.30 | 1878.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 1853.00 | 1848.36 | 1859.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 1853.00 | 1848.36 | 1859.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1853.00 | 1848.36 | 1859.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 1850.90 | 1848.36 | 1859.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1853.80 | 1848.24 | 1856.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:15:00 | 1849.00 | 1848.24 | 1856.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:00:00 | 1849.10 | 1848.41 | 1856.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:30:00 | 1847.00 | 1846.04 | 1852.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:45:00 | 1847.50 | 1850.95 | 1853.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 1859.50 | 1852.66 | 1854.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 1859.00 | 1852.66 | 1854.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 1870.30 | 1856.19 | 1855.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 1870.30 | 1856.19 | 1855.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 15:15:00 | 1884.90 | 1861.93 | 1858.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 14:15:00 | 1842.10 | 1861.93 | 1860.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 14:15:00 | 1842.10 | 1861.93 | 1860.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1842.10 | 1861.93 | 1860.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 1842.10 | 1861.93 | 1860.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 1843.20 | 1858.19 | 1859.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 1837.60 | 1852.13 | 1855.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1884.70 | 1855.04 | 1855.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1884.70 | 1855.04 | 1855.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1884.70 | 1855.04 | 1855.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 1884.70 | 1855.04 | 1855.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 10:15:00 | 1875.00 | 1859.03 | 1857.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1897.70 | 1877.24 | 1868.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1942.70 | 1944.53 | 1929.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:45:00 | 1942.50 | 1944.53 | 1929.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1940.00 | 1943.90 | 1934.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 1935.80 | 1943.90 | 1934.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 1938.40 | 1942.80 | 1935.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 1937.00 | 1942.80 | 1935.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1945.00 | 1943.24 | 1936.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 1926.20 | 1943.24 | 1936.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1965.40 | 1967.95 | 1958.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:30:00 | 1955.00 | 1967.95 | 1958.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1967.90 | 1968.06 | 1960.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:45:00 | 1963.80 | 1968.06 | 1960.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1970.80 | 1968.12 | 1961.79 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1962.20 | 1969.00 | 1969.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1946.10 | 1964.42 | 1966.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1943.00 | 1936.09 | 1947.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 1943.00 | 1936.09 | 1947.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1936.50 | 1936.18 | 1946.63 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 15:15:00 | 1951.10 | 1947.10 | 1946.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1961.00 | 1951.11 | 1948.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 12:15:00 | 1947.00 | 1951.95 | 1949.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 12:15:00 | 1947.00 | 1951.95 | 1949.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1947.00 | 1951.95 | 1949.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 1947.00 | 1951.95 | 1949.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1951.80 | 1951.92 | 1949.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 1963.50 | 1954.65 | 1951.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 10:45:00 | 1959.90 | 1962.88 | 1959.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 1946.00 | 1961.69 | 1963.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 1946.00 | 1961.69 | 1963.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 1941.00 | 1956.23 | 1960.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 1952.00 | 1949.39 | 1955.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 1952.00 | 1949.39 | 1955.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1952.00 | 1949.39 | 1955.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 1955.40 | 1949.39 | 1955.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1954.10 | 1950.65 | 1954.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1954.10 | 1950.65 | 1954.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1949.90 | 1950.50 | 1954.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:30:00 | 1947.80 | 1951.68 | 1954.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 1958.00 | 1952.95 | 1954.85 | SL hit (close>static) qty=1.00 sl=1955.70 alert=retest2 |

### Cycle 114 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1977.80 | 1957.92 | 1956.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2012.10 | 1980.52 | 1969.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 2046.00 | 2047.04 | 2019.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:00:00 | 2046.00 | 2047.04 | 2019.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 2025.40 | 2042.77 | 2026.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 2023.20 | 2042.77 | 2026.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 2019.30 | 2038.08 | 2025.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 2019.30 | 2038.08 | 2025.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 2021.80 | 2034.82 | 2025.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 2026.60 | 2034.82 | 2025.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 2007.00 | 2022.14 | 2022.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 2007.00 | 2022.14 | 2022.19 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 2037.00 | 2016.12 | 2015.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 2042.80 | 2028.67 | 2022.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 2021.50 | 2029.95 | 2025.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 2021.50 | 2029.95 | 2025.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 2021.50 | 2029.95 | 2025.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 2024.30 | 2029.95 | 2025.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 2027.00 | 2029.36 | 2025.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 2036.20 | 2027.21 | 2025.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 2060.40 | 2086.83 | 2087.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 2060.40 | 2086.83 | 2087.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 2038.60 | 2068.57 | 2078.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 2016.90 | 2012.42 | 2034.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 2019.00 | 2012.42 | 2034.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1961.90 | 2003.13 | 2026.56 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 2021.50 | 2015.77 | 2015.40 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 2008.00 | 2014.25 | 2014.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 1996.80 | 2010.76 | 2013.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1952.90 | 1946.43 | 1964.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 1952.90 | 1946.43 | 1964.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1985.50 | 1950.55 | 1960.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 1985.50 | 1950.55 | 1960.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1978.10 | 1956.06 | 1962.00 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 1978.20 | 1967.19 | 1966.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1992.20 | 1972.19 | 1968.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 1982.50 | 1987.94 | 1978.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 1982.50 | 1987.94 | 1978.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1975.20 | 1985.72 | 1979.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 1975.20 | 1985.72 | 1979.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1966.80 | 1981.94 | 1978.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 1966.80 | 1981.94 | 1978.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1968.70 | 1979.29 | 1977.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1987.10 | 1979.29 | 1977.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 1982.40 | 1994.80 | 1995.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1982.40 | 1994.80 | 1995.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 1971.30 | 1990.10 | 1993.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1914.10 | 1906.73 | 1938.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 1914.10 | 1906.73 | 1938.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1922.20 | 1909.83 | 1936.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1975.50 | 1909.83 | 1936.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1961.60 | 1920.18 | 1938.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:30:00 | 1954.60 | 1946.47 | 1947.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 1953.40 | 1947.85 | 1947.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 1953.40 | 1947.85 | 1947.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1987.40 | 1957.31 | 1952.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 1955.70 | 1958.41 | 1954.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 12:15:00 | 1955.70 | 1958.41 | 1954.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1955.70 | 1958.41 | 1954.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:30:00 | 1955.30 | 1958.41 | 1954.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1965.00 | 1959.73 | 1955.08 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 1929.50 | 1952.98 | 1953.14 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 1974.30 | 1956.42 | 1953.98 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 1930.20 | 1949.30 | 1951.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 1918.80 | 1943.20 | 1948.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 1966.20 | 1942.23 | 1944.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 1966.20 | 1942.23 | 1944.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1966.20 | 1942.23 | 1944.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 1966.20 | 1942.23 | 1944.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1976.50 | 1949.09 | 1947.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 1986.90 | 1970.06 | 1964.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 2010.30 | 2020.64 | 2005.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 10:00:00 | 2010.30 | 2020.64 | 2005.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 2038.30 | 2024.17 | 2008.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 2056.80 | 2029.16 | 2026.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:30:00 | 2046.60 | 2032.47 | 2028.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 2006.70 | 2025.09 | 2026.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 2006.70 | 2025.09 | 2026.17 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 2046.70 | 2028.28 | 2027.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 2054.90 | 2042.63 | 2037.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 2120.00 | 2132.00 | 2113.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 2120.00 | 2132.00 | 2113.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2120.00 | 2132.00 | 2113.90 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 2052.00 | 2106.45 | 2108.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 2041.50 | 2093.46 | 2102.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 2066.50 | 2060.49 | 2078.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 2066.50 | 2060.49 | 2078.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 2066.50 | 2060.49 | 2078.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 2087.20 | 2060.49 | 2078.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 2082.40 | 2064.87 | 2078.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 2082.40 | 2064.87 | 2078.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 2081.20 | 2068.14 | 2078.92 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 2101.10 | 2085.31 | 2084.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 2130.90 | 2094.43 | 2088.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 2066.00 | 2104.64 | 2098.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 2066.00 | 2104.64 | 2098.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2066.00 | 2104.64 | 2098.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:15:00 | 2057.50 | 2104.64 | 2098.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 2072.30 | 2098.18 | 2096.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:30:00 | 2061.20 | 2098.18 | 2096.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 2077.30 | 2094.00 | 2094.79 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 13:15:00 | 2105.80 | 2095.91 | 2095.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 14:15:00 | 2120.20 | 2100.77 | 2097.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 2225.00 | 2248.63 | 2211.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:30:00 | 2225.00 | 2248.63 | 2211.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 2186.60 | 2238.33 | 2224.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 2186.60 | 2238.33 | 2224.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 2179.00 | 2226.46 | 2220.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 2168.00 | 2226.46 | 2220.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 2180.50 | 2209.85 | 2213.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 2171.00 | 2198.67 | 2207.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 2177.10 | 2174.24 | 2188.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:30:00 | 2181.60 | 2174.24 | 2188.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 2209.50 | 2167.04 | 2173.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 2209.50 | 2167.04 | 2173.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 2211.70 | 2175.97 | 2177.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 2211.70 | 2175.97 | 2177.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 2232.10 | 2187.20 | 2182.39 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 2151.30 | 2182.54 | 2182.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 2129.30 | 2171.89 | 2177.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2140.80 | 2132.50 | 2151.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 2140.80 | 2132.50 | 2151.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2140.80 | 2132.50 | 2151.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 2144.90 | 2132.50 | 2151.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 2165.80 | 2139.16 | 2152.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 2165.80 | 2139.16 | 2152.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 2170.70 | 2145.47 | 2154.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:45:00 | 2167.90 | 2145.47 | 2154.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 2180.40 | 2163.10 | 2161.32 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 2122.50 | 2157.69 | 2159.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 2115.10 | 2149.17 | 2155.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 2112.60 | 2111.79 | 2130.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 2112.60 | 2111.79 | 2130.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 2112.60 | 2111.79 | 2130.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 2101.00 | 2111.79 | 2130.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:00:00 | 2096.20 | 2101.42 | 2120.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 2146.30 | 2107.50 | 2116.83 | SL hit (close>static) qty=1.00 sl=2137.10 alert=retest2 |

### Cycle 138 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 2164.50 | 2126.31 | 2124.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 2168.80 | 2139.64 | 2130.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 2152.20 | 2161.99 | 2151.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 2152.20 | 2161.99 | 2151.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 2152.20 | 2161.99 | 2151.99 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 2122.00 | 2146.50 | 2148.45 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 2196.80 | 2156.56 | 2152.85 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 13:15:00 | 2109.80 | 2143.14 | 2147.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 14:15:00 | 2100.00 | 2134.51 | 2143.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 2080.70 | 2077.69 | 2104.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 2080.70 | 2077.69 | 2104.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 2090.50 | 2081.62 | 2096.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:45:00 | 2091.10 | 2081.62 | 2096.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 2100.00 | 2085.29 | 2096.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:15:00 | 2108.90 | 2085.29 | 2096.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 2100.00 | 2088.24 | 2097.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:30:00 | 2100.00 | 2088.24 | 2097.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 2103.30 | 2091.25 | 2097.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:45:00 | 2108.70 | 2091.25 | 2097.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 2122.00 | 2102.27 | 2101.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 2139.40 | 2119.89 | 2111.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 2161.30 | 2163.98 | 2147.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 2158.40 | 2163.98 | 2147.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 2174.70 | 2168.79 | 2155.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:45:00 | 2162.00 | 2168.79 | 2155.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 2167.90 | 2168.50 | 2157.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 13:30:00 | 2176.40 | 2168.80 | 2158.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 2145.20 | 2162.36 | 2157.97 | SL hit (close<static) qty=1.00 sl=2155.00 alert=retest2 |

### Cycle 143 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 2231.40 | 2236.09 | 2236.41 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 2241.50 | 2237.17 | 2236.88 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 2227.80 | 2235.33 | 2236.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 12:15:00 | 2227.30 | 2233.73 | 2235.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 14:15:00 | 2233.00 | 2232.57 | 2234.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 14:15:00 | 2233.00 | 2232.57 | 2234.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 2233.00 | 2232.57 | 2234.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:15:00 | 2235.00 | 2232.57 | 2234.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 2235.00 | 2233.06 | 2234.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 2218.50 | 2233.06 | 2234.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 2240.90 | 2228.79 | 2230.73 | SL hit (close>static) qty=1.00 sl=2237.90 alert=retest2 |

### Cycle 146 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 2352.00 | 2254.48 | 2242.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 2396.70 | 2329.13 | 2309.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 2397.70 | 2410.22 | 2388.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 11:00:00 | 2397.70 | 2410.22 | 2388.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 2403.60 | 2406.17 | 2390.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:45:00 | 2410.00 | 2407.00 | 2392.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 2425.20 | 2405.37 | 2394.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 2409.20 | 2405.83 | 2396.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 2388.00 | 2396.25 | 2395.25 | SL hit (close<static) qty=1.00 sl=2389.30 alert=retest2 |

### Cycle 147 — SELL (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 15:15:00 | 2379.00 | 2399.77 | 2402.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 2349.70 | 2373.31 | 2385.37 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-16 13:15:00 | 989.00 | 2024-05-17 10:15:00 | 1018.00 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-06-03 13:00:00 | 1172.20 | 2024-06-04 09:15:00 | 1143.70 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-06-13 10:15:00 | 1214.40 | 2024-06-21 14:15:00 | 1229.15 | STOP_HIT | 1.00 | 1.21% |
| BUY | retest2 | 2024-06-13 11:30:00 | 1213.50 | 2024-06-21 14:15:00 | 1229.15 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2024-06-13 12:15:00 | 1213.40 | 2024-06-21 14:15:00 | 1229.15 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-07-01 09:15:00 | 1238.05 | 2024-07-08 09:15:00 | 1361.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-27 10:15:00 | 1714.85 | 2024-08-29 15:15:00 | 1686.05 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-09-03 11:30:00 | 1704.05 | 2024-09-05 11:15:00 | 1713.90 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-09-13 11:45:00 | 1749.90 | 2024-09-17 09:15:00 | 1728.05 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-09-13 13:30:00 | 1750.30 | 2024-09-17 09:15:00 | 1728.05 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-09-16 09:15:00 | 1751.25 | 2024-09-17 09:15:00 | 1728.05 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-09-16 10:30:00 | 1750.70 | 2024-09-17 09:15:00 | 1728.05 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-09-20 10:15:00 | 1632.00 | 2024-09-23 09:15:00 | 1744.70 | STOP_HIT | 1.00 | -6.91% |
| SELL | retest2 | 2024-09-20 14:00:00 | 1637.70 | 2024-09-23 09:15:00 | 1744.70 | STOP_HIT | 1.00 | -6.53% |
| SELL | retest2 | 2024-09-20 14:30:00 | 1630.90 | 2024-09-23 09:15:00 | 1744.70 | STOP_HIT | 1.00 | -6.98% |
| SELL | retest2 | 2024-09-30 15:00:00 | 1671.25 | 2024-10-04 11:15:00 | 1693.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-10-01 09:15:00 | 1669.35 | 2024-10-04 11:15:00 | 1693.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-10-03 09:45:00 | 1668.80 | 2024-10-04 11:15:00 | 1693.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-10-21 11:30:00 | 1717.20 | 2024-10-28 13:15:00 | 1714.60 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2024-10-22 10:15:00 | 1710.25 | 2024-10-28 13:15:00 | 1714.60 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-11-05 13:15:00 | 1699.00 | 2024-11-07 11:15:00 | 1672.15 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-11-18 14:00:00 | 1507.05 | 2024-11-25 13:15:00 | 1495.95 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2024-11-19 11:45:00 | 1506.05 | 2024-11-25 13:15:00 | 1495.95 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2024-11-19 14:15:00 | 1498.75 | 2024-11-25 13:15:00 | 1495.95 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-12-23 11:15:00 | 1556.15 | 2024-12-24 14:15:00 | 1534.75 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-12-24 10:45:00 | 1556.50 | 2024-12-24 14:15:00 | 1534.75 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-12-24 11:15:00 | 1557.90 | 2024-12-24 14:15:00 | 1534.75 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-01-02 11:45:00 | 1624.45 | 2025-01-08 12:15:00 | 1606.20 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-01-06 11:15:00 | 1616.95 | 2025-01-08 12:15:00 | 1606.20 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-01-06 11:45:00 | 1616.00 | 2025-01-08 12:15:00 | 1606.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-01-17 09:15:00 | 1440.15 | 2025-01-20 12:15:00 | 1500.60 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2025-01-17 10:45:00 | 1446.60 | 2025-01-20 12:15:00 | 1500.60 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-01-20 09:15:00 | 1450.40 | 2025-01-20 12:15:00 | 1500.60 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2025-01-23 10:15:00 | 1517.05 | 2025-01-24 09:15:00 | 1465.60 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-01-23 12:00:00 | 1519.55 | 2025-01-24 09:15:00 | 1465.60 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-02-14 10:15:00 | 1348.00 | 2025-02-17 09:15:00 | 1280.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:15:00 | 1348.00 | 2025-02-17 09:15:00 | 1372.30 | STOP_HIT | 0.50 | -1.80% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1342.60 | 2025-02-28 09:15:00 | 1280.93 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2025-02-20 12:15:00 | 1348.35 | 2025-02-28 09:15:00 | 1281.26 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1342.60 | 2025-03-03 10:15:00 | 1292.95 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2025-02-20 12:15:00 | 1348.35 | 2025-03-03 10:15:00 | 1292.95 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-02-20 14:15:00 | 1348.70 | 2025-03-04 09:15:00 | 1314.60 | STOP_HIT | 1.00 | 2.53% |
| SELL | retest2 | 2025-02-24 13:30:00 | 1334.30 | 2025-03-04 09:15:00 | 1314.60 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-03-13 14:15:00 | 1408.50 | 2025-03-25 14:15:00 | 1479.25 | STOP_HIT | 1.00 | 5.02% |
| BUY | retest2 | 2025-03-17 09:15:00 | 1427.40 | 2025-03-25 14:15:00 | 1479.25 | STOP_HIT | 1.00 | 3.63% |
| BUY | retest2 | 2025-04-02 10:45:00 | 1525.20 | 2025-04-04 09:15:00 | 1475.00 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-04-02 12:00:00 | 1523.75 | 2025-04-04 09:15:00 | 1475.00 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-04-02 14:00:00 | 1522.00 | 2025-04-04 09:15:00 | 1475.00 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-04-03 09:15:00 | 1563.30 | 2025-04-04 09:15:00 | 1475.00 | STOP_HIT | 1.00 | -5.65% |
| SELL | retest1 | 2025-04-15 11:45:00 | 1369.80 | 2025-04-21 11:15:00 | 1360.20 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest1 | 2025-04-16 09:45:00 | 1370.00 | 2025-04-21 11:15:00 | 1360.20 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2025-04-17 10:30:00 | 1348.90 | 2025-04-22 09:15:00 | 1386.70 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-04-17 11:45:00 | 1345.40 | 2025-04-22 09:15:00 | 1386.70 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-05-02 13:45:00 | 1376.00 | 2025-05-05 13:15:00 | 1401.50 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-05-05 13:15:00 | 1377.00 | 2025-05-05 13:15:00 | 1401.50 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-05-07 12:15:00 | 1425.00 | 2025-05-09 10:15:00 | 1398.40 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-05-16 12:15:00 | 1454.00 | 2025-05-19 14:15:00 | 1432.10 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-05-19 09:45:00 | 1464.80 | 2025-05-19 14:15:00 | 1432.10 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-05-19 11:45:00 | 1455.10 | 2025-05-19 14:15:00 | 1432.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-05-19 12:15:00 | 1457.90 | 2025-05-19 14:15:00 | 1432.10 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-05-23 10:45:00 | 1413.20 | 2025-05-29 15:15:00 | 1410.80 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-05-23 12:30:00 | 1413.20 | 2025-05-29 15:15:00 | 1410.80 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-05-26 09:45:00 | 1413.00 | 2025-05-29 15:15:00 | 1410.80 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-06-11 12:00:00 | 1626.40 | 2025-06-18 11:15:00 | 1657.90 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2025-06-26 10:00:00 | 1739.00 | 2025-07-09 12:15:00 | 1806.60 | STOP_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2025-06-26 11:15:00 | 1729.70 | 2025-07-09 12:15:00 | 1806.60 | STOP_HIT | 1.00 | 4.45% |
| BUY | retest2 | 2025-06-26 12:45:00 | 1730.00 | 2025-07-09 12:15:00 | 1806.60 | STOP_HIT | 1.00 | 4.43% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1732.10 | 2025-07-09 12:15:00 | 1806.60 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2025-06-27 15:15:00 | 1720.00 | 2025-07-09 12:15:00 | 1806.60 | STOP_HIT | 1.00 | 5.03% |
| SELL | retest2 | 2025-07-28 09:45:00 | 2140.00 | 2025-07-28 10:15:00 | 2154.90 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-08-08 09:15:00 | 2042.60 | 2025-08-11 14:15:00 | 2049.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-08-21 13:15:00 | 1940.60 | 2025-08-25 12:15:00 | 1960.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-08-21 14:30:00 | 1940.70 | 2025-08-25 12:15:00 | 1960.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-08-22 10:45:00 | 1941.80 | 2025-08-25 12:15:00 | 1960.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-22 12:00:00 | 1936.40 | 2025-08-25 12:15:00 | 1960.90 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-02 14:45:00 | 1914.50 | 2025-09-03 09:15:00 | 1965.90 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-09-24 14:45:00 | 2017.30 | 2025-10-03 12:15:00 | 1983.90 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1983.00 | 2025-10-03 12:15:00 | 1983.90 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-10-09 13:45:00 | 1934.70 | 2025-10-10 11:15:00 | 1964.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-10-09 15:00:00 | 1935.10 | 2025-10-10 11:15:00 | 1964.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-10-10 10:15:00 | 1929.40 | 2025-10-10 11:15:00 | 1964.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1930.00 | 2025-10-24 11:15:00 | 1833.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 10:15:00 | 1907.70 | 2025-10-24 14:15:00 | 1812.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 11:00:00 | 1904.10 | 2025-10-24 14:15:00 | 1812.03 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-10-13 12:30:00 | 1903.50 | 2025-10-27 09:15:00 | 1808.89 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-10-14 09:15:00 | 1907.40 | 2025-10-27 09:15:00 | 1808.32 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2025-10-14 10:45:00 | 1890.00 | 2025-10-27 09:15:00 | 1804.81 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2025-10-14 14:30:00 | 1899.80 | 2025-10-27 09:15:00 | 1804.43 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-10-14 15:00:00 | 1899.40 | 2025-10-27 09:15:00 | 1803.29 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-10-15 13:15:00 | 1898.20 | 2025-10-28 13:15:00 | 1795.50 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1930.00 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 5.53% |
| SELL | retest2 | 2025-10-13 10:15:00 | 1907.70 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2025-10-13 11:00:00 | 1904.10 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2025-10-13 12:30:00 | 1903.50 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2025-10-14 09:15:00 | 1907.40 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2025-10-14 10:45:00 | 1890.00 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2025-10-14 14:30:00 | 1899.80 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-10-14 15:00:00 | 1899.40 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2025-10-15 13:15:00 | 1898.20 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2025-10-23 13:45:00 | 1851.40 | 2025-10-29 12:15:00 | 1839.00 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-10-23 15:00:00 | 1847.00 | 2025-10-29 12:15:00 | 1839.00 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-11-10 13:30:00 | 1832.90 | 2025-11-12 09:15:00 | 1838.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-11-10 15:00:00 | 1829.60 | 2025-11-12 09:15:00 | 1838.10 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-11-11 09:15:00 | 1830.80 | 2025-11-12 10:15:00 | 1851.70 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-11 10:15:00 | 1819.60 | 2025-11-12 10:15:00 | 1851.70 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-11-11 14:00:00 | 1817.90 | 2025-11-12 10:15:00 | 1851.70 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-11-11 15:00:00 | 1816.70 | 2025-11-12 10:15:00 | 1851.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-11-12 09:45:00 | 1815.10 | 2025-11-12 10:15:00 | 1851.70 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-11-14 11:00:00 | 1910.00 | 2025-11-17 15:15:00 | 1867.30 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-11-14 12:15:00 | 1923.20 | 2025-11-17 15:15:00 | 1867.30 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2025-11-17 10:00:00 | 1917.00 | 2025-11-17 15:15:00 | 1867.30 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-11-19 13:15:00 | 1849.00 | 2025-11-20 14:15:00 | 1870.30 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-19 14:00:00 | 1849.10 | 2025-11-20 14:15:00 | 1870.30 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-20 09:30:00 | 1847.00 | 2025-11-20 14:15:00 | 1870.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-11-20 12:45:00 | 1847.50 | 2025-11-20 14:15:00 | 1870.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-12 09:45:00 | 1963.50 | 2025-12-17 10:15:00 | 1946.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-15 10:45:00 | 1959.90 | 2025-12-17 10:15:00 | 1946.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-12-18 14:30:00 | 1947.80 | 2025-12-18 15:15:00 | 1958.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-12-24 12:15:00 | 2026.60 | 2025-12-26 09:15:00 | 2007.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-02 10:15:00 | 2036.20 | 2026-01-09 09:15:00 | 2060.40 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2026-01-27 09:15:00 | 1987.10 | 2026-02-01 11:15:00 | 1982.40 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2026-02-03 13:30:00 | 1954.60 | 2026-02-03 14:15:00 | 1953.40 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2026-02-19 11:45:00 | 2056.80 | 2026-02-19 15:15:00 | 2006.70 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-02-19 12:30:00 | 2046.60 | 2026-02-19 15:15:00 | 2006.70 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-03-24 10:15:00 | 2101.00 | 2026-03-25 09:15:00 | 2146.30 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-03-24 13:00:00 | 2096.20 | 2026-03-25 09:15:00 | 2146.30 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-04-10 13:30:00 | 2176.40 | 2026-04-13 09:15:00 | 2145.20 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-04-13 11:45:00 | 2176.50 | 2026-04-20 15:15:00 | 2231.40 | STOP_HIT | 1.00 | 2.52% |
| SELL | retest2 | 2026-04-22 09:15:00 | 2218.50 | 2026-04-22 14:15:00 | 2240.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-04-30 13:45:00 | 2410.00 | 2026-05-05 09:15:00 | 2388.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-05-04 09:15:00 | 2425.20 | 2026-05-05 09:15:00 | 2388.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-05-04 11:15:00 | 2409.20 | 2026-05-05 09:15:00 | 2388.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-05-05 09:30:00 | 2411.40 | 2026-05-06 14:15:00 | 2362.20 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-05-05 11:15:00 | 2409.20 | 2026-05-06 14:15:00 | 2362.20 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-05-05 13:00:00 | 2416.00 | 2026-05-06 14:15:00 | 2362.20 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-05-06 09:45:00 | 2417.20 | 2026-05-06 14:15:00 | 2362.20 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-05-06 10:45:00 | 2413.80 | 2026-05-06 14:15:00 | 2362.20 | STOP_HIT | 1.00 | -2.14% |
