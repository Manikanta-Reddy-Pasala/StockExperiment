# Brigade Enterprises Ltd. (BRIGADE)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 760.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 1 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 50 |
| PARTIAL | 2 |
| TARGET_HIT | 14 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 36
- **Target hits / Stop hits / Partials:** 14 / 36 / 2
- **Avg / median % per leg:** 0.94% / -1.77%
- **Sum % (uncompounded):** 48.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 8 | 33.3% | 8 | 16 | 0 | 1.51% | 36.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 8 | 33.3% | 8 | 16 | 0 | 1.51% | 36.2% |
| SELL (all) | 28 | 8 | 28.6% | 6 | 20 | 2 | 0.45% | 12.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 8 | 28.6% | 6 | 20 | 2 | 0.45% | 12.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 16 | 30.8% | 14 | 36 | 2 | 0.94% | 48.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 1122.60 | 1224.97 | 1225.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 10:15:00 | 1117.10 | 1222.94 | 1224.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 15:15:00 | 1192.50 | 1190.64 | 1205.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-28 09:15:00 | 1177.55 | 1190.64 | 1205.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1200.00 | 1190.06 | 1204.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 1200.00 | 1190.06 | 1204.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 1201.95 | 1190.44 | 1204.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 1201.95 | 1190.44 | 1204.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1205.05 | 1190.59 | 1204.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 1203.10 | 1190.59 | 1204.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1206.20 | 1190.74 | 1204.11 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 12:15:00 | 1321.00 | 1214.60 | 1214.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1342.05 | 1242.83 | 1229.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 1323.40 | 1328.57 | 1291.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 1323.40 | 1328.57 | 1291.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 1288.65 | 1326.79 | 1292.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:00:00 | 1288.65 | 1326.79 | 1292.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 1286.15 | 1326.38 | 1291.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 15:15:00 | 1299.80 | 1326.05 | 1291.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 10:15:00 | 1270.55 | 1324.84 | 1291.88 | SL hit (close<static) qty=1.00 sl=1283.70 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 1141.55 | 1273.10 | 1273.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 1132.35 | 1241.54 | 1255.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1200.80 | 1194.31 | 1224.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 1200.80 | 1194.31 | 1224.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 1231.10 | 1195.37 | 1223.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 1231.10 | 1195.37 | 1223.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 1262.90 | 1196.04 | 1223.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 1218.25 | 1196.04 | 1223.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 1228.80 | 1197.98 | 1223.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 12:45:00 | 1229.15 | 1198.29 | 1223.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 1224.25 | 1201.75 | 1224.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1219.35 | 1201.92 | 1224.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-03 09:15:00 | 1267.65 | 1207.44 | 1225.39 | SL hit (close>static) qty=1.00 sl=1265.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 1273.90 | 1239.47 | 1239.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 1294.20 | 1240.33 | 1239.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 1236.20 | 1250.20 | 1245.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 1236.20 | 1250.20 | 1245.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1236.20 | 1250.20 | 1245.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 1236.20 | 1250.20 | 1245.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1245.00 | 1250.15 | 1245.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 1255.95 | 1250.15 | 1245.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 09:45:00 | 1258.20 | 1251.66 | 1246.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 15:00:00 | 1246.90 | 1251.82 | 1246.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 09:15:00 | 1247.45 | 1251.68 | 1246.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1241.25 | 1251.58 | 1246.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:15:00 | 1245.75 | 1251.58 | 1246.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1250.00 | 1251.56 | 1246.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 13:15:00 | 1258.85 | 1251.54 | 1246.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 11:15:00 | 1232.00 | 1250.90 | 1246.45 | SL hit (close<static) qty=1.00 sl=1237.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 1155.65 | 1244.39 | 1244.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 1146.15 | 1243.41 | 1243.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 11:15:00 | 1159.75 | 1140.62 | 1180.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 12:00:00 | 1159.75 | 1140.62 | 1180.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1180.00 | 1142.14 | 1179.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 1180.00 | 1142.14 | 1179.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1167.60 | 1142.39 | 1179.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 13:00:00 | 1140.20 | 1142.37 | 1179.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1147.10 | 1142.46 | 1179.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 1213.50 | 1143.61 | 1179.51 | SL hit (close>static) qty=1.00 sl=1184.70 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 14:15:00 | 1109.30 | 1026.04 | 1025.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 15:15:00 | 1118.20 | 1026.96 | 1026.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 1148.60 | 1149.83 | 1106.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:45:00 | 1152.40 | 1149.83 | 1106.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 1108.60 | 1147.51 | 1114.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 1108.60 | 1147.51 | 1114.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1116.70 | 1147.21 | 1114.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 1119.20 | 1123.56 | 1107.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1107.40 | 1123.40 | 1107.67 | SL hit (close<static) qty=1.00 sl=1108.30 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1032.00 | 1099.07 | 1099.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 1021.10 | 1098.29 | 1098.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 963.90 | 957.36 | 994.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:00:00 | 963.90 | 957.36 | 994.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 944.45 | 927.63 | 959.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 947.40 | 927.63 | 959.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 958.55 | 928.56 | 956.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 958.55 | 928.56 | 956.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 958.50 | 928.85 | 956.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:30:00 | 959.00 | 928.85 | 956.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 952.00 | 929.08 | 956.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 969.50 | 929.08 | 956.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 967.05 | 929.46 | 956.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 970.40 | 929.46 | 956.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 965.15 | 929.82 | 956.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 965.15 | 929.82 | 956.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1040.60 | 973.15 | 973.04 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 10:15:00 | 944.90 | 974.40 | 974.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 13:15:00 | 944.30 | 973.52 | 973.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 11:15:00 | 891.20 | 888.57 | 911.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 12:00:00 | 891.20 | 888.57 | 911.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 718.35 | 690.46 | 738.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 744.25 | 690.46 | 738.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 724.90 | 686.53 | 727.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:30:00 | 724.70 | 686.53 | 727.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 724.00 | 686.90 | 727.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:45:00 | 727.90 | 686.90 | 727.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 727.90 | 687.31 | 727.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 12:00:00 | 727.90 | 687.31 | 727.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 720.25 | 687.64 | 727.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 15:00:00 | 717.60 | 688.26 | 727.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 710.25 | 688.59 | 727.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:00:00 | 717.05 | 689.94 | 726.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 712.50 | 691.90 | 726.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 725.00 | 692.23 | 726.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:00:00 | 725.00 | 692.23 | 726.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 727.75 | 692.59 | 726.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 727.75 | 692.59 | 726.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 735.95 | 693.02 | 726.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 735.95 | 693.02 | 726.41 | SL hit (close>static) qty=1.00 sl=729.10 alert=retest2 |

### Cycle 10 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 792.90 | 746.03 | 745.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 801.45 | 750.65 | 748.21 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-10 11:45:00 | 574.00 | 2023-09-04 09:15:00 | 631.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-10 12:45:00 | 574.05 | 2023-09-04 09:15:00 | 631.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-10 13:30:00 | 580.05 | 2023-09-04 09:15:00 | 638.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-22 10:15:00 | 575.50 | 2023-09-28 12:15:00 | 569.75 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-09-27 09:15:00 | 594.10 | 2023-09-28 12:15:00 | 569.75 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2023-09-27 09:45:00 | 593.80 | 2023-10-18 15:15:00 | 633.05 | TARGET_HIT | 1.00 | 6.61% |
| BUY | retest2 | 2023-10-06 12:30:00 | 593.85 | 2023-11-03 11:15:00 | 653.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-06 15:15:00 | 593.20 | 2023-11-03 11:15:00 | 652.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-09 12:15:00 | 594.20 | 2023-11-03 11:15:00 | 653.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-26 13:15:00 | 596.70 | 2023-11-03 12:15:00 | 656.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-11 15:15:00 | 1299.80 | 2024-10-14 10:15:00 | 1270.55 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-10-15 14:00:00 | 1308.90 | 2024-10-17 13:15:00 | 1281.45 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-10-22 09:15:00 | 1302.30 | 2024-10-22 09:15:00 | 1229.15 | STOP_HIT | 1.00 | -5.62% |
| SELL | retest2 | 2024-11-26 09:15:00 | 1218.25 | 2024-12-03 09:15:00 | 1267.65 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2024-11-27 12:15:00 | 1228.80 | 2024-12-03 09:15:00 | 1267.65 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-11-27 12:45:00 | 1229.15 | 2024-12-03 09:15:00 | 1267.65 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-11-29 09:15:00 | 1224.25 | 2024-12-03 09:15:00 | 1267.65 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-12-23 09:15:00 | 1255.95 | 2024-12-30 11:15:00 | 1232.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-12-24 09:45:00 | 1258.20 | 2025-01-07 11:15:00 | 1234.35 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-12-26 15:00:00 | 1246.90 | 2025-01-07 11:15:00 | 1234.35 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-12-27 09:15:00 | 1247.45 | 2025-01-09 11:15:00 | 1212.35 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-12-27 13:15:00 | 1258.85 | 2025-01-09 11:15:00 | 1212.35 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2025-01-02 11:15:00 | 1255.10 | 2025-01-09 11:15:00 | 1212.35 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2025-01-06 15:15:00 | 1257.50 | 2025-01-09 11:15:00 | 1212.35 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-02-01 13:00:00 | 1140.20 | 2025-02-03 09:15:00 | 1213.50 | STOP_HIT | 1.00 | -6.43% |
| SELL | retest2 | 2025-02-01 14:15:00 | 1147.10 | 2025-02-03 09:15:00 | 1213.50 | STOP_HIT | 1.00 | -5.79% |
| SELL | retest2 | 2025-02-06 09:45:00 | 1144.20 | 2025-02-11 09:15:00 | 1086.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 12:45:00 | 1146.35 | 2025-02-11 09:15:00 | 1089.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 09:45:00 | 1144.20 | 2025-02-12 10:15:00 | 1029.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 12:45:00 | 1146.35 | 2025-02-12 10:15:00 | 1031.71 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 10:15:00 | 988.20 | 2025-04-07 09:15:00 | 889.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-26 09:30:00 | 986.95 | 2025-04-07 09:15:00 | 888.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-26 10:15:00 | 985.95 | 2025-04-07 09:15:00 | 887.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-28 09:15:00 | 989.70 | 2025-04-07 09:15:00 | 890.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-25 09:45:00 | 1012.85 | 2025-04-28 09:15:00 | 1039.20 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-04-25 11:15:00 | 1010.90 | 2025-04-28 09:15:00 | 1039.20 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-04-29 12:00:00 | 1012.50 | 2025-05-07 15:15:00 | 1023.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-04-29 15:15:00 | 1012.80 | 2025-05-07 15:15:00 | 1023.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-05-02 11:30:00 | 1007.20 | 2025-05-07 15:15:00 | 1023.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-05-06 11:30:00 | 1005.20 | 2025-05-07 15:15:00 | 1023.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-05-06 13:30:00 | 1007.40 | 2025-05-09 15:15:00 | 1022.90 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-05-07 13:15:00 | 1007.60 | 2025-05-09 15:15:00 | 1022.90 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-05-08 13:15:00 | 1010.50 | 2025-05-12 10:15:00 | 1037.00 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-05-09 14:15:00 | 1010.70 | 2025-05-12 10:15:00 | 1037.00 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-07-08 09:15:00 | 1119.20 | 2025-07-08 09:15:00 | 1107.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-08 13:15:00 | 1118.50 | 2025-07-09 09:15:00 | 1101.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-16 09:30:00 | 1119.40 | 2025-07-21 09:15:00 | 1097.30 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-16 10:00:00 | 1124.60 | 2025-07-21 09:15:00 | 1097.30 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-04-08 15:00:00 | 717.60 | 2026-04-13 11:15:00 | 735.95 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-04-09 09:15:00 | 710.25 | 2026-04-13 11:15:00 | 735.95 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2026-04-10 10:00:00 | 717.05 | 2026-04-13 11:15:00 | 735.95 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-04-13 09:15:00 | 712.50 | 2026-04-13 11:15:00 | 735.95 | STOP_HIT | 1.00 | -3.29% |
