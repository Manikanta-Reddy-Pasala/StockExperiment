# Jyoti CNC Automation Ltd. (JYOTICNC)

## Backtest Summary

- **Window:** 2024-01-16 09:15:00 → 2026-05-11 15:15:00 (3993 bars)
- **Last close:** 749.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 56 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 18
- **Target hits / Stop hits / Partials:** 1 / 18 / 0
- **Avg / median % per leg:** -2.36% / -1.91%
- **Sum % (uncompounded):** -44.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.92% | -23.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.92% | -23.1% |
| SELL (all) | 7 | 1 | 14.3% | 1 | 6 | 0 | -3.10% | -21.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 1 | 6 | 0 | -3.10% | -21.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 1 | 5.3% | 1 | 18 | 0 | -2.36% | -44.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 1116.00 | 1142.69 | 1142.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 15:15:00 | 1113.00 | 1141.44 | 1142.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 1108.60 | 1086.12 | 1108.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 1108.60 | 1086.12 | 1108.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1108.60 | 1086.12 | 1108.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 1116.60 | 1086.12 | 1108.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1119.20 | 1086.45 | 1108.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 1119.20 | 1086.45 | 1108.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 1119.45 | 1086.78 | 1108.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:30:00 | 1120.05 | 1086.78 | 1108.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 1114.90 | 1087.39 | 1108.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 1114.90 | 1087.39 | 1108.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 1118.00 | 1087.69 | 1108.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 1139.55 | 1087.69 | 1108.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1155.00 | 1088.99 | 1109.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 1155.00 | 1088.99 | 1109.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 1116.00 | 1096.93 | 1111.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 1103.10 | 1096.93 | 1111.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1103.10 | 1096.99 | 1111.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:30:00 | 1091.30 | 1096.98 | 1111.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 1092.10 | 1094.56 | 1108.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 11:45:00 | 1094.55 | 1094.59 | 1108.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 12:15:00 | 1194.00 | 1095.58 | 1108.61 | SL hit (close>static) qty=1.00 sl=1122.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 1226.40 | 1121.20 | 1120.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 14:15:00 | 1250.00 | 1127.58 | 1123.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 14:15:00 | 1315.30 | 1320.26 | 1266.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 14:45:00 | 1318.95 | 1320.26 | 1266.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1265.10 | 1319.58 | 1266.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 1265.10 | 1319.58 | 1266.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 1279.15 | 1319.18 | 1266.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 1265.10 | 1319.18 | 1266.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 1264.65 | 1317.77 | 1266.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:00:00 | 1264.65 | 1317.77 | 1266.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 1253.15 | 1317.12 | 1266.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 15:00:00 | 1253.15 | 1317.12 | 1266.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 1250.70 | 1316.46 | 1266.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:15:00 | 1217.50 | 1316.46 | 1266.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1211.10 | 1315.41 | 1266.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:30:00 | 1208.05 | 1315.41 | 1266.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 1248.50 | 1291.43 | 1258.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 11:30:00 | 1254.35 | 1291.43 | 1258.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 1238.80 | 1290.91 | 1258.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:30:00 | 1238.25 | 1290.91 | 1258.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1253.20 | 1284.50 | 1257.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 1253.20 | 1284.50 | 1257.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 1265.85 | 1284.31 | 1257.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:15:00 | 1269.95 | 1284.31 | 1257.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:45:00 | 1272.25 | 1284.14 | 1257.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 13:15:00 | 1269.90 | 1284.14 | 1257.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 15:00:00 | 1269.20 | 1283.80 | 1257.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1257.90 | 1283.70 | 1257.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1257.90 | 1283.70 | 1257.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1260.10 | 1283.47 | 1257.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 1260.05 | 1283.47 | 1257.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 1259.00 | 1283.23 | 1257.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 1258.80 | 1283.23 | 1257.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 1255.60 | 1282.95 | 1257.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:00:00 | 1255.60 | 1282.95 | 1257.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 1247.50 | 1282.60 | 1257.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-21 13:15:00 | 1247.50 | 1282.60 | 1257.77 | SL hit (close<static) qty=1.00 sl=1253.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 988.70 | 1237.01 | 1237.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 953.70 | 1131.79 | 1172.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 14:15:00 | 970.25 | 965.72 | 1054.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 970.25 | 965.72 | 1054.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1042.00 | 969.49 | 1043.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 1042.00 | 969.49 | 1043.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 1038.75 | 970.18 | 1043.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 1044.40 | 970.18 | 1043.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1048.45 | 972.87 | 1042.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 1042.90 | 972.87 | 1042.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1050.05 | 973.64 | 1042.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 1053.65 | 973.64 | 1042.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 1045.60 | 975.76 | 1042.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:30:00 | 1039.55 | 1013.54 | 1050.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 1040.80 | 1013.86 | 1050.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 1050.30 | 1014.22 | 1050.75 | SL hit (close>static) qty=1.00 sl=1049.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 1145.00 | 1059.06 | 1058.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 1165.00 | 1064.41 | 1061.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 1206.50 | 1215.72 | 1163.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 12:00:00 | 1206.50 | 1215.72 | 1163.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1174.90 | 1216.93 | 1170.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 1174.90 | 1216.93 | 1170.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1165.00 | 1215.19 | 1170.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1165.00 | 1215.19 | 1170.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1170.00 | 1214.74 | 1170.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 1163.70 | 1214.74 | 1170.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1151.60 | 1214.11 | 1170.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1151.60 | 1214.11 | 1170.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 1052.50 | 1145.52 | 1145.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 1042.80 | 1143.58 | 1144.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 13:15:00 | 1076.60 | 1076.54 | 1103.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:45:00 | 1080.80 | 1076.54 | 1103.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 983.10 | 917.61 | 949.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 983.10 | 917.61 | 949.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1005.75 | 918.48 | 949.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 1005.75 | 918.48 | 949.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 929.45 | 906.49 | 930.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:45:00 | 931.30 | 906.49 | 930.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 942.80 | 906.86 | 930.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 942.80 | 906.86 | 930.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 941.75 | 907.20 | 930.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 943.15 | 907.20 | 930.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 924.95 | 906.56 | 927.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:15:00 | 949.55 | 906.56 | 927.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 954.25 | 907.04 | 927.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 951.25 | 907.04 | 927.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 967.20 | 907.63 | 927.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:45:00 | 963.25 | 907.63 | 927.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 952.00 | 916.47 | 930.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 952.00 | 916.47 | 930.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 943.00 | 916.73 | 930.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 969.60 | 916.73 | 930.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 1008.55 | 942.12 | 941.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 1037.85 | 943.85 | 942.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 964.20 | 969.90 | 957.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:00:00 | 964.20 | 969.90 | 957.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 962.00 | 969.75 | 957.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 967.00 | 969.45 | 957.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:30:00 | 971.60 | 969.46 | 957.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 946.30 | 968.29 | 958.03 | SL hit (close<static) qty=1.00 sl=956.20 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 916.15 | 957.24 | 957.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 910.50 | 956.77 | 957.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 10:15:00 | 871.65 | 866.65 | 899.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 871.65 | 866.65 | 899.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 811.10 | 776.42 | 811.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:00:00 | 811.10 | 776.42 | 811.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 806.85 | 776.72 | 811.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:45:00 | 806.05 | 776.72 | 811.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 810.00 | 777.38 | 811.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:15:00 | 825.60 | 777.38 | 811.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 815.15 | 777.75 | 811.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 699.60 | 783.16 | 811.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-12 12:30:00 | 1091.30 | 2024-11-19 12:15:00 | 1194.00 | STOP_HIT | 1.00 | -9.41% |
| SELL | retest2 | 2024-11-19 09:15:00 | 1092.10 | 2024-11-19 12:15:00 | 1194.00 | STOP_HIT | 1.00 | -9.33% |
| SELL | retest2 | 2024-11-19 11:45:00 | 1094.55 | 2024-11-19 12:15:00 | 1194.00 | STOP_HIT | 1.00 | -9.09% |
| BUY | retest2 | 2025-01-20 12:15:00 | 1269.95 | 2025-01-21 13:15:00 | 1247.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-01-20 12:45:00 | 1272.25 | 2025-01-21 13:15:00 | 1247.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-01-20 13:15:00 | 1269.90 | 2025-01-21 13:15:00 | 1247.50 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-01-20 15:00:00 | 1269.20 | 2025-01-21 13:15:00 | 1247.50 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-03-26 14:30:00 | 1039.55 | 2025-03-27 09:15:00 | 1050.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-03-27 09:15:00 | 1040.80 | 2025-03-27 09:15:00 | 1050.30 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-04-01 09:15:00 | 1034.05 | 2025-04-01 09:15:00 | 1053.75 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-04-04 11:00:00 | 1040.50 | 2025-04-07 09:15:00 | 936.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-03 15:15:00 | 967.00 | 2025-12-05 14:15:00 | 946.30 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-12-04 09:30:00 | 971.60 | 2025-12-05 14:15:00 | 946.30 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-12-10 09:15:00 | 974.80 | 2025-12-10 15:15:00 | 940.30 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2025-12-11 14:15:00 | 975.90 | 2025-12-12 15:15:00 | 954.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-12-22 14:30:00 | 962.30 | 2026-01-08 09:15:00 | 954.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-23 09:15:00 | 966.30 | 2026-01-08 09:15:00 | 954.10 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-24 09:15:00 | 977.30 | 2026-01-08 09:15:00 | 954.10 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-01-07 14:15:00 | 962.65 | 2026-01-08 09:15:00 | 954.10 | STOP_HIT | 1.00 | -0.89% |
