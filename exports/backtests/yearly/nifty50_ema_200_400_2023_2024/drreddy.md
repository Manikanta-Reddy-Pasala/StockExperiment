# DRREDDY (DRREDDY)

## Backtest Summary

- **Window:** 2022-04-07 09:15:00 → 2026-05-08 15:15:00 (7054 bars)
- **Last close:** 1294.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 22 |
| ALERT1 | 15 |
| ALERT2 | 15 |
| ALERT2_SKIP | 8 |
| ALERT3 | 84 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 95 |
| PARTIAL | 6 |
| TARGET_HIT | 7 |
| STOP_HIT | 89 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 101 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 81
- **Target hits / Stop hits / Partials:** 7 / 88 / 6
- **Avg / median % per leg:** -0.10% / -0.99%
- **Sum % (uncompounded):** -9.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 8 | 12.1% | 5 | 61 | 0 | -0.53% | -35.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 66 | 8 | 12.1% | 5 | 61 | 0 | -0.53% | -35.1% |
| SELL (all) | 35 | 12 | 34.3% | 2 | 27 | 6 | 0.72% | 25.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 12 | 34.3% | 2 | 27 | 6 | 0.72% | 25.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 101 | 20 | 19.8% | 7 | 88 | 6 | -0.10% | -9.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-06-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 15:15:00 | 910.00 | 919.66 | 919.68 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 14:15:00 | 922.69 | 919.73 | 919.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 926.36 | 919.83 | 919.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 13:15:00 | 1135.04 | 1135.51 | 1086.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 13:30:00 | 1134.59 | 1135.51 | 1086.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 1110.58 | 1133.29 | 1105.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 09:15:00 | 1112.82 | 1127.16 | 1104.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 13:15:00 | 1098.20 | 1126.12 | 1104.83 | SL hit (close<static) qty=1.00 sl=1100.40 alert=retest2 |

### Cycle 3 — SELL (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 09:15:00 | 1063.85 | 1099.40 | 1099.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 10:15:00 | 1060.87 | 1099.02 | 1099.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 1097.78 | 1094.09 | 1096.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 1097.78 | 1094.09 | 1096.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 1097.78 | 1094.09 | 1096.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 10:00:00 | 1097.78 | 1094.09 | 1096.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 1096.37 | 1094.11 | 1096.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:00:00 | 1096.37 | 1094.11 | 1096.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 1097.00 | 1094.14 | 1096.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:30:00 | 1095.51 | 1094.14 | 1096.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 12:15:00 | 1092.00 | 1094.12 | 1096.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-08 14:30:00 | 1089.34 | 1094.08 | 1096.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-08 15:15:00 | 1088.62 | 1094.08 | 1096.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-09 10:15:00 | 1098.63 | 1094.09 | 1096.56 | SL hit (close>static) qty=1.00 sl=1097.40 alert=retest2 |

### Cycle 4 — BUY (started 2023-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 12:15:00 | 1130.80 | 1098.07 | 1097.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 1141.00 | 1099.46 | 1098.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 09:15:00 | 1093.18 | 1126.32 | 1114.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 1093.18 | 1126.32 | 1114.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 1093.18 | 1126.32 | 1114.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 09:45:00 | 1120.46 | 1124.32 | 1114.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 10:00:00 | 1115.80 | 1123.38 | 1114.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 10:45:00 | 1116.49 | 1123.33 | 1114.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 13:30:00 | 1116.66 | 1123.10 | 1114.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 14:15:00 | 1114.20 | 1122.80 | 1114.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 14:45:00 | 1114.47 | 1122.80 | 1114.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 15:15:00 | 1115.80 | 1122.73 | 1114.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 09:15:00 | 1119.91 | 1122.73 | 1114.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 11:00:00 | 1116.56 | 1122.67 | 1114.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-15 11:15:00 | 1113.07 | 1122.57 | 1114.46 | SL hit (close<static) qty=1.00 sl=1113.60 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 13:15:00 | 1183.08 | 1222.11 | 1222.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 14:15:00 | 1173.60 | 1221.62 | 1221.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 1199.84 | 1197.55 | 1207.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 13:00:00 | 1199.84 | 1197.55 | 1207.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 1204.56 | 1186.78 | 1199.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:30:00 | 1204.00 | 1186.78 | 1199.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 1209.75 | 1187.01 | 1199.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 12:30:00 | 1213.20 | 1187.01 | 1199.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 1212.22 | 1191.88 | 1200.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:30:00 | 1213.02 | 1191.88 | 1200.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1202.35 | 1195.60 | 1201.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 13:30:00 | 1198.95 | 1195.75 | 1201.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1208.00 | 1195.29 | 1201.13 | SL hit (close>static) qty=1.00 sl=1203.76 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 14:15:00 | 1279.44 | 1205.97 | 1205.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 1284.52 | 1207.48 | 1206.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 1370.01 | 1371.93 | 1336.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 1370.01 | 1371.93 | 1336.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 1339.03 | 1369.95 | 1338.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 1339.03 | 1369.95 | 1338.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1337.57 | 1369.34 | 1338.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 09:30:00 | 1345.60 | 1362.33 | 1337.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 1330.00 | 1361.48 | 1337.49 | SL hit (close<static) qty=1.00 sl=1330.60 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 1286.35 | 1331.81 | 1331.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 1272.70 | 1331.22 | 1331.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 1320.60 | 1311.15 | 1320.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 10:15:00 | 1320.60 | 1311.15 | 1320.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1320.60 | 1311.15 | 1320.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:45:00 | 1319.35 | 1311.15 | 1320.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 1301.45 | 1311.05 | 1320.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:30:00 | 1294.95 | 1310.48 | 1319.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 12:15:00 | 1230.20 | 1296.88 | 1311.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-06 11:15:00 | 1251.60 | 1245.52 | 1273.40 | SL hit (close>ema200) qty=0.50 sl=1245.52 alert=retest2 |

### Cycle 8 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 1389.70 | 1283.63 | 1283.34 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 12:15:00 | 1182.00 | 1299.55 | 1300.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 09:15:00 | 1172.25 | 1238.86 | 1261.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 1164.55 | 1162.32 | 1202.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 12:00:00 | 1164.55 | 1162.32 | 1202.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 1196.10 | 1165.14 | 1199.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:00:00 | 1196.10 | 1165.14 | 1199.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 1197.15 | 1165.46 | 1199.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 14:45:00 | 1193.00 | 1165.82 | 1199.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 15:15:00 | 1202.25 | 1166.18 | 1199.47 | SL hit (close>static) qty=1.00 sl=1200.50 alert=retest2 |

### Cycle 10 — BUY (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 11:15:00 | 1232.10 | 1182.48 | 1182.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1240.90 | 1184.73 | 1183.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 1278.40 | 1294.54 | 1259.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:00:00 | 1278.40 | 1294.54 | 1259.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1256.90 | 1290.18 | 1265.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1256.90 | 1290.18 | 1265.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1257.30 | 1289.85 | 1265.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 1253.90 | 1289.85 | 1265.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1269.00 | 1289.40 | 1265.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:15:00 | 1272.10 | 1280.11 | 1264.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 14:15:00 | 1263.50 | 1279.74 | 1264.22 | SL hit (close<static) qty=1.00 sl=1263.90 alert=retest2 |

### Cycle 11 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 1188.80 | 1257.75 | 1257.90 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1272.80 | 1255.51 | 1255.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1274.40 | 1255.70 | 1255.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1256.70 | 1257.31 | 1256.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 1256.70 | 1257.31 | 1256.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1256.70 | 1257.31 | 1256.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 1256.70 | 1257.31 | 1256.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1250.00 | 1257.23 | 1256.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1252.80 | 1257.23 | 1256.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1257.60 | 1257.24 | 1256.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:30:00 | 1264.70 | 1257.30 | 1256.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 12:00:00 | 1262.10 | 1257.30 | 1256.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 15:00:00 | 1262.00 | 1257.47 | 1256.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 1263.30 | 1257.88 | 1256.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1257.50 | 1258.38 | 1257.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 1257.50 | 1258.38 | 1257.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 1256.10 | 1258.36 | 1257.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:45:00 | 1254.20 | 1258.36 | 1257.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1259.40 | 1258.37 | 1257.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 1261.10 | 1258.37 | 1257.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:45:00 | 1261.40 | 1258.37 | 1257.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 14:15:00 | 1250.80 | 1258.30 | 1257.08 | SL hit (close<static) qty=1.00 sl=1254.20 alert=retest2 |

### Cycle 13 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1251.50 | 1265.94 | 1265.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1250.10 | 1265.79 | 1265.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 1272.00 | 1259.68 | 1262.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1277.80 | 1259.86 | 1262.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 1277.80 | 1259.86 | 1262.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1280.90 | 1264.32 | 1264.69 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1288.80 | 1265.08 | 1265.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 1293.90 | 1266.73 | 1265.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 1252.00 | 1267.38 | 1266.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1254.70 | 1267.25 | 1266.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 1243.50 | 1267.25 | 1266.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1199.10 | 1265.09 | 1265.13 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1266.00 | 1252.56 | 1252.55 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1247.10 | 1252.53 | 1252.54 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 1259.60 | 1252.60 | 1252.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1261.40 | 1252.69 | 1252.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 1259.50 | 1263.96 | 1259.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1260.10 | 1263.92 | 1259.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 1265.30 | 1263.94 | 1259.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 1265.80 | 1263.92 | 1259.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:30:00 | 1266.70 | 1263.94 | 1259.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1267.20 | 1263.98 | 1259.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1265.10 | 1264.36 | 1259.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 1270.20 | 1264.38 | 1259.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:45:00 | 1269.90 | 1264.44 | 1260.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 1254.10 | 1264.54 | 1260.18 | SL hit (close<static) qty=1.00 sl=1255.50 alert=retest2 |

### Cycle 19 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1210.20 | 1256.67 | 1256.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1204.20 | 1254.40 | 1255.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 1243.50 | 1225.87 | 1239.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1243.00 | 1226.04 | 1239.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 1246.00 | 1226.04 | 1239.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1247.70 | 1226.38 | 1239.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1247.70 | 1226.38 | 1239.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1232.00 | 1226.54 | 1239.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 1238.70 | 1226.54 | 1239.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1238.40 | 1226.66 | 1239.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 1242.90 | 1226.66 | 1239.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1245.30 | 1226.85 | 1239.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:30:00 | 1235.70 | 1227.32 | 1239.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:15:00 | 1235.10 | 1227.32 | 1239.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:30:00 | 1231.60 | 1227.57 | 1239.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 15:15:00 | 1173.91 | 1224.08 | 1235.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 15:15:00 | 1173.34 | 1224.08 | 1235.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 15:15:00 | 1170.02 | 1224.08 | 1235.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1221.90 | 1221.12 | 1233.83 | SL hit (close>ema200) qty=0.50 sl=1221.12 alert=retest2 |

### Cycle 20 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1286.00 | 1241.91 | 1241.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1288.00 | 1245.82 | 1243.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 1283.30 | 1283.55 | 1268.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 13:00:00 | 1283.30 | 1283.55 | 1268.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1274.60 | 1283.70 | 1268.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 1269.00 | 1283.70 | 1268.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 1271.30 | 1283.58 | 1268.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 11:45:00 | 1267.10 | 1283.58 | 1268.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 1268.20 | 1283.43 | 1268.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 1268.20 | 1283.43 | 1268.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 1273.50 | 1283.33 | 1268.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:30:00 | 1271.20 | 1283.33 | 1268.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1268.40 | 1282.96 | 1268.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:45:00 | 1267.70 | 1282.96 | 1268.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 1279.00 | 1282.92 | 1268.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:30:00 | 1268.10 | 1282.92 | 1268.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1276.30 | 1283.64 | 1269.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1298.00 | 1283.03 | 1269.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.29 | 1270.93 | SL hit (close<static) qty=1.00 sl=1266.70 alert=retest2 |

### Cycle 21 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 1187.90 | 1263.24 | 1263.25 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1340.40 | 1258.95 | 1258.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1359.00 | 1259.94 | 1259.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1273.80 | 1274.75 | 1267.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 12:30:00 | 1272.10 | 1274.75 | 1267.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-12 11:15:00 | 899.00 | 2023-06-01 15:15:00 | 910.00 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2023-05-15 09:30:00 | 898.76 | 2023-06-01 15:15:00 | 910.00 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2023-05-24 09:15:00 | 902.00 | 2023-06-01 15:15:00 | 910.00 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2023-09-27 09:15:00 | 1112.82 | 2023-09-27 13:15:00 | 1098.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2023-09-29 10:30:00 | 1115.20 | 2023-10-03 10:15:00 | 1099.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2023-10-11 09:15:00 | 1111.60 | 2023-10-13 11:15:00 | 1096.91 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2023-10-12 11:30:00 | 1111.82 | 2023-10-13 11:15:00 | 1096.91 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-10-13 11:00:00 | 1102.83 | 2023-10-26 11:15:00 | 1080.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2023-10-17 09:15:00 | 1105.25 | 2023-10-26 11:15:00 | 1080.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2023-10-25 15:00:00 | 1103.50 | 2023-10-26 11:15:00 | 1080.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2023-11-08 14:30:00 | 1089.34 | 2023-11-09 10:15:00 | 1098.63 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2023-11-08 15:15:00 | 1088.62 | 2023-11-09 10:15:00 | 1098.63 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-11-09 14:30:00 | 1089.65 | 2023-11-16 11:15:00 | 1097.25 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2023-11-10 09:15:00 | 1078.00 | 2023-11-16 11:15:00 | 1097.25 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2023-11-13 11:00:00 | 1083.32 | 2023-11-16 11:15:00 | 1097.25 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-11-15 09:30:00 | 1084.54 | 2023-11-16 11:15:00 | 1097.25 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2023-11-15 12:15:00 | 1085.67 | 2023-11-16 12:15:00 | 1101.53 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2023-11-15 14:45:00 | 1083.79 | 2023-11-16 12:15:00 | 1101.53 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2023-12-12 09:45:00 | 1120.46 | 2023-12-15 11:15:00 | 1113.07 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2023-12-13 10:00:00 | 1115.80 | 2023-12-15 11:15:00 | 1113.07 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2023-12-13 10:45:00 | 1116.49 | 2023-12-21 09:15:00 | 1106.60 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-12-13 13:30:00 | 1116.66 | 2023-12-21 13:15:00 | 1109.37 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-12-15 09:15:00 | 1119.91 | 2024-01-31 11:15:00 | 1227.38 | TARGET_HIT | 1.00 | 9.60% |
| BUY | retest2 | 2023-12-15 11:00:00 | 1116.56 | 2024-02-01 09:15:00 | 1228.14 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2023-12-15 15:15:00 | 1120.76 | 2024-02-01 09:15:00 | 1228.33 | TARGET_HIT | 1.00 | 9.60% |
| BUY | retest2 | 2023-12-21 12:00:00 | 1117.52 | 2024-02-06 09:15:00 | 1232.51 | TARGET_HIT | 1.00 | 10.29% |
| BUY | retest2 | 2024-01-24 09:15:00 | 1148.97 | 2024-02-12 09:15:00 | 1263.87 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-18 13:30:00 | 1198.95 | 2024-06-21 09:15:00 | 1208.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-06-21 15:15:00 | 1198.40 | 2024-06-24 14:15:00 | 1206.61 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-09-11 09:30:00 | 1345.60 | 2024-09-11 12:15:00 | 1330.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-09-26 10:00:00 | 1345.57 | 2024-10-04 13:15:00 | 1330.03 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-09-26 10:30:00 | 1345.87 | 2024-10-04 13:15:00 | 1330.03 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-09-26 12:15:00 | 1344.09 | 2024-10-04 13:15:00 | 1330.03 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-09-27 12:30:00 | 1351.80 | 2024-10-04 13:15:00 | 1330.03 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-09-27 13:30:00 | 1351.80 | 2024-10-04 13:15:00 | 1330.03 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-09-27 14:30:00 | 1348.87 | 2024-10-04 13:15:00 | 1330.03 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-09-30 11:30:00 | 1353.40 | 2024-10-04 13:15:00 | 1330.03 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-10-04 12:15:00 | 1340.60 | 2024-10-04 13:15:00 | 1330.03 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-10-04 13:00:00 | 1339.05 | 2024-10-04 13:15:00 | 1330.03 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-10-07 09:30:00 | 1339.12 | 2024-10-07 11:15:00 | 1331.13 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-10-09 09:15:00 | 1345.25 | 2024-10-09 15:15:00 | 1332.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-10-09 11:15:00 | 1339.19 | 2024-10-10 09:15:00 | 1325.73 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-10-09 12:00:00 | 1340.08 | 2024-10-10 09:15:00 | 1325.73 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-10-16 12:30:00 | 1342.44 | 2024-10-22 14:15:00 | 1331.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-10-16 14:00:00 | 1344.02 | 2024-10-22 14:15:00 | 1331.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-10-17 12:15:00 | 1341.00 | 2024-10-22 14:15:00 | 1331.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-10-17 15:00:00 | 1341.65 | 2024-10-22 14:15:00 | 1331.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-10-18 09:45:00 | 1345.20 | 2024-10-22 14:15:00 | 1331.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-10-21 10:30:00 | 1341.00 | 2024-10-22 14:15:00 | 1331.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-11-07 09:30:00 | 1294.95 | 2024-11-14 12:15:00 | 1230.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 09:30:00 | 1294.95 | 2024-12-06 11:15:00 | 1251.60 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2025-03-21 14:45:00 | 1193.00 | 2025-03-21 15:15:00 | 1202.25 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-03-24 09:30:00 | 1191.15 | 2025-03-24 10:15:00 | 1202.35 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-03-25 09:15:00 | 1177.95 | 2025-04-04 09:15:00 | 1119.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:30:00 | 1189.65 | 2025-04-04 09:15:00 | 1130.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 09:15:00 | 1177.95 | 2025-04-07 09:15:00 | 1060.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 09:30:00 | 1189.65 | 2025-04-07 09:15:00 | 1070.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-22 09:15:00 | 1166.50 | 2025-04-23 09:15:00 | 1182.40 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-04-22 12:30:00 | 1175.70 | 2025-04-23 09:15:00 | 1182.40 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-04-25 10:15:00 | 1171.50 | 2025-04-28 09:15:00 | 1202.20 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-04-25 14:00:00 | 1171.30 | 2025-04-28 09:15:00 | 1202.20 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-04-29 12:15:00 | 1170.60 | 2025-04-30 09:15:00 | 1189.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-05-05 10:30:00 | 1164.10 | 2025-05-12 09:15:00 | 1183.20 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1155.30 | 2025-05-12 09:15:00 | 1183.20 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-07-17 12:15:00 | 1272.10 | 2025-07-17 14:15:00 | 1263.50 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-24 09:15:00 | 1284.00 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2025-07-24 10:00:00 | 1276.90 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-07-25 09:15:00 | 1283.00 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-07-28 14:30:00 | 1289.00 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2025-07-29 09:45:00 | 1288.40 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2025-07-29 10:30:00 | 1288.20 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2025-07-29 12:00:00 | 1290.10 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2025-07-31 11:30:00 | 1280.50 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-07-31 12:45:00 | 1280.00 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-09-03 11:30:00 | 1264.70 | 2025-09-08 14:15:00 | 1250.80 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-09-03 12:00:00 | 1262.10 | 2025-09-08 14:15:00 | 1250.80 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-03 15:00:00 | 1262.00 | 2025-09-26 14:15:00 | 1252.90 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-05 09:30:00 | 1263.30 | 2025-09-26 14:15:00 | 1252.90 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-09-08 13:15:00 | 1261.10 | 2025-09-29 11:15:00 | 1244.60 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-08 13:45:00 | 1261.40 | 2025-09-29 11:15:00 | 1244.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-09 09:15:00 | 1274.80 | 2025-09-29 11:15:00 | 1244.60 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-09-26 09:30:00 | 1267.40 | 2025-09-29 11:15:00 | 1244.60 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-10-10 14:30:00 | 1264.10 | 2025-10-13 09:15:00 | 1251.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-24 12:00:00 | 1265.30 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-24 12:45:00 | 1265.80 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-12-24 14:30:00 | 1266.70 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-12-26 10:15:00 | 1267.20 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-12-31 12:00:00 | 1270.20 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-12-31 12:45:00 | 1269.90 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-27 13:30:00 | 1235.70 | 2026-02-01 15:15:00 | 1173.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 14:15:00 | 1235.10 | 2026-02-01 15:15:00 | 1173.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 09:30:00 | 1231.60 | 2026-02-01 15:15:00 | 1170.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 13:30:00 | 1235.70 | 2026-02-03 09:15:00 | 1221.90 | STOP_HIT | 0.50 | 1.12% |
| SELL | retest2 | 2026-01-27 14:15:00 | 1235.10 | 2026-02-03 09:15:00 | 1221.90 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2026-01-28 09:30:00 | 1231.60 | 2026-02-03 09:15:00 | 1221.90 | STOP_HIT | 0.50 | 0.79% |
| SELL | retest2 | 2026-02-03 12:15:00 | 1235.20 | 2026-02-04 09:15:00 | 1243.80 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-02-03 15:15:00 | 1228.90 | 2026-02-06 14:15:00 | 1240.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-02-06 09:30:00 | 1230.60 | 2026-02-06 14:15:00 | 1240.90 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-06 12:15:00 | 1229.40 | 2026-02-09 09:15:00 | 1263.70 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1298.00 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2026-03-25 09:45:00 | 1287.80 | 2026-03-30 09:15:00 | 1260.40 | STOP_HIT | 1.00 | -2.13% |
