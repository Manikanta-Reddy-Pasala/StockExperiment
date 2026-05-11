# SBI Life Insurance Company Ltd. (SBILIFE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1871.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 236 |
| ALERT1 | 157 |
| ALERT2 | 155 |
| ALERT2_SKIP | 81 |
| ALERT3 | 408 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 196 |
| PARTIAL | 6 |
| TARGET_HIT | 9 |
| STOP_HIT | 196 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 211 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 72 / 139
- **Target hits / Stop hits / Partials:** 9 / 196 / 6
- **Avg / median % per leg:** 0.20% / -0.70%
- **Sum % (uncompounded):** 43.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 96 | 36 | 37.5% | 9 | 87 | 0 | 0.61% | 58.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.06% | -0.1% |
| BUY @ 3rd Alert (retest2) | 95 | 36 | 37.9% | 9 | 86 | 0 | 0.62% | 58.8% |
| SELL (all) | 115 | 36 | 31.3% | 0 | 109 | 6 | -0.14% | -15.7% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 8 | 0 | 1.53% | 12.2% |
| SELL @ 3rd Alert (retest2) | 107 | 28 | 26.2% | 0 | 101 | 6 | -0.26% | -27.9% |
| retest1 (combined) | 9 | 8 | 88.9% | 0 | 9 | 0 | 1.35% | 12.2% |
| retest2 (combined) | 202 | 64 | 31.7% | 9 | 187 | 6 | 0.15% | 30.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 09:15:00 | 1175.85 | 1185.31 | 1186.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 1165.80 | 1179.51 | 1183.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 14:15:00 | 1155.30 | 1151.38 | 1158.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-19 15:00:00 | 1155.30 | 1151.38 | 1158.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 1158.85 | 1154.21 | 1158.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:45:00 | 1157.70 | 1154.21 | 1158.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 11:15:00 | 1155.95 | 1154.56 | 1158.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 12:30:00 | 1151.65 | 1154.37 | 1157.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 09:15:00 | 1158.95 | 1154.57 | 1156.69 | SL hit (close>static) qty=1.00 sl=1158.80 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 12:15:00 | 1165.75 | 1158.56 | 1158.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 15:15:00 | 1166.45 | 1162.16 | 1160.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 09:15:00 | 1205.85 | 1210.19 | 1199.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 09:30:00 | 1204.45 | 1210.19 | 1199.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 12:15:00 | 1217.00 | 1225.99 | 1220.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 13:00:00 | 1217.00 | 1225.99 | 1220.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 13:15:00 | 1209.00 | 1222.59 | 1219.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 14:00:00 | 1209.00 | 1222.59 | 1219.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-06-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 15:15:00 | 1210.00 | 1217.69 | 1217.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 14:15:00 | 1202.05 | 1209.67 | 1213.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 09:15:00 | 1211.80 | 1209.11 | 1212.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 09:15:00 | 1211.80 | 1209.11 | 1212.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 1211.80 | 1209.11 | 1212.25 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 09:15:00 | 1215.50 | 1212.96 | 1212.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 15:15:00 | 1224.50 | 1219.22 | 1216.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 14:15:00 | 1237.00 | 1243.24 | 1236.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 14:15:00 | 1237.00 | 1243.24 | 1236.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 1237.00 | 1243.24 | 1236.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 1237.00 | 1243.24 | 1236.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 1242.00 | 1242.99 | 1237.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:15:00 | 1249.95 | 1242.99 | 1237.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 1251.70 | 1244.73 | 1238.42 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 09:15:00 | 1236.70 | 1241.31 | 1241.51 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 09:15:00 | 1249.40 | 1242.30 | 1241.73 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 15:15:00 | 1240.00 | 1241.30 | 1241.47 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 09:15:00 | 1253.90 | 1243.82 | 1242.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 10:15:00 | 1255.95 | 1246.25 | 1243.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 15:15:00 | 1277.50 | 1279.37 | 1269.88 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:15:00 | 1291.70 | 1279.37 | 1269.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 1298.95 | 1296.42 | 1290.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-22 12:15:00 | 1290.95 | 1295.74 | 1291.63 | SL hit (close<ema400) qty=1.00 sl=1291.63 alert=retest1 |

### Cycle 9 — SELL (started 2023-06-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 15:15:00 | 1276.65 | 1287.96 | 1288.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 1270.00 | 1284.37 | 1287.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 1271.30 | 1270.54 | 1277.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 10:15:00 | 1270.20 | 1270.54 | 1277.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 1261.15 | 1267.05 | 1272.11 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 15:15:00 | 1287.50 | 1275.07 | 1273.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 10:15:00 | 1298.20 | 1281.83 | 1277.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 10:15:00 | 1298.45 | 1304.07 | 1297.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 10:15:00 | 1298.45 | 1304.07 | 1297.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 10:15:00 | 1298.45 | 1304.07 | 1297.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:00:00 | 1298.45 | 1304.07 | 1297.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 1292.65 | 1301.79 | 1297.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 12:00:00 | 1292.65 | 1301.79 | 1297.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 12:15:00 | 1295.85 | 1300.60 | 1297.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 13:30:00 | 1296.15 | 1300.00 | 1297.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 12:00:00 | 1296.55 | 1298.09 | 1297.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-04 13:15:00 | 1283.35 | 1294.30 | 1295.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 13:15:00 | 1283.35 | 1294.30 | 1295.69 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 1313.15 | 1299.08 | 1297.37 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 11:15:00 | 1290.25 | 1297.63 | 1297.98 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 10:15:00 | 1315.85 | 1300.36 | 1298.62 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 09:15:00 | 1286.95 | 1296.30 | 1297.34 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 1302.45 | 1296.66 | 1296.11 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 15:15:00 | 1292.15 | 1295.83 | 1296.04 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 1297.60 | 1296.19 | 1296.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 1320.10 | 1304.27 | 1300.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 1315.85 | 1316.68 | 1308.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 14:00:00 | 1315.85 | 1316.68 | 1308.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 1316.70 | 1316.42 | 1311.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:30:00 | 1314.90 | 1316.42 | 1311.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 1312.45 | 1317.10 | 1313.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:30:00 | 1311.20 | 1317.10 | 1313.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 1303.25 | 1314.33 | 1312.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:00:00 | 1303.25 | 1314.33 | 1312.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 11:15:00 | 1302.50 | 1311.96 | 1311.98 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 15:15:00 | 1315.00 | 1312.47 | 1312.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 09:15:00 | 1319.65 | 1313.90 | 1312.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 1308.40 | 1313.62 | 1312.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 11:15:00 | 1308.40 | 1313.62 | 1312.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 1308.40 | 1313.62 | 1312.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:00:00 | 1308.40 | 1313.62 | 1312.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 1311.15 | 1313.12 | 1312.77 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 13:15:00 | 1309.80 | 1312.43 | 1312.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 09:15:00 | 1307.50 | 1310.75 | 1311.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 11:15:00 | 1310.50 | 1309.78 | 1311.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 11:15:00 | 1310.50 | 1309.78 | 1311.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 1310.50 | 1309.78 | 1311.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:00:00 | 1310.50 | 1309.78 | 1311.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 1314.45 | 1310.72 | 1311.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:00:00 | 1314.45 | 1310.72 | 1311.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 1315.00 | 1311.57 | 1311.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 14:00:00 | 1315.00 | 1311.57 | 1311.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-07-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 14:15:00 | 1314.90 | 1312.24 | 1312.08 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 10:15:00 | 1309.70 | 1311.69 | 1311.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 13:15:00 | 1305.05 | 1309.57 | 1310.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 1304.45 | 1303.74 | 1307.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-24 10:00:00 | 1304.45 | 1303.74 | 1307.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 1313.45 | 1305.68 | 1307.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 11:00:00 | 1313.45 | 1305.68 | 1307.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 1307.00 | 1305.94 | 1307.82 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 13:15:00 | 1316.20 | 1309.33 | 1309.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 14:15:00 | 1317.55 | 1310.98 | 1309.89 | Break + close above crossover candle high |

### Cycle 25 — SELL (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 09:15:00 | 1296.20 | 1308.82 | 1309.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 09:15:00 | 1285.65 | 1298.64 | 1303.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 1305.00 | 1295.52 | 1298.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 1305.00 | 1295.52 | 1298.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 1305.00 | 1295.52 | 1298.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 09:15:00 | 1283.80 | 1296.78 | 1297.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 10:15:00 | 1291.95 | 1296.52 | 1297.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 10:45:00 | 1285.90 | 1294.52 | 1296.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 10:15:00 | 1279.90 | 1268.13 | 1267.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 10:15:00 | 1279.90 | 1268.13 | 1267.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 11:15:00 | 1285.00 | 1271.50 | 1269.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 12:15:00 | 1342.70 | 1347.44 | 1334.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 13:00:00 | 1342.70 | 1347.44 | 1334.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 1335.80 | 1343.58 | 1335.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 15:00:00 | 1335.80 | 1343.58 | 1335.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 1331.60 | 1341.19 | 1334.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:15:00 | 1325.75 | 1341.19 | 1334.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 1320.05 | 1336.96 | 1333.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:45:00 | 1320.15 | 1336.96 | 1333.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 1319.60 | 1333.49 | 1332.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:30:00 | 1318.85 | 1333.49 | 1332.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 11:15:00 | 1309.40 | 1328.67 | 1330.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 13:15:00 | 1309.05 | 1321.76 | 1326.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 14:15:00 | 1284.55 | 1284.52 | 1292.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-17 15:00:00 | 1284.55 | 1284.52 | 1292.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 1280.00 | 1278.43 | 1283.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 09:15:00 | 1276.55 | 1278.43 | 1283.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 10:15:00 | 1288.40 | 1280.30 | 1283.79 | SL hit (close>static) qty=1.00 sl=1285.85 alert=retest2 |

### Cycle 28 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 1304.60 | 1285.43 | 1284.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 10:15:00 | 1307.65 | 1289.88 | 1286.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 13:15:00 | 1291.25 | 1292.31 | 1288.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 14:00:00 | 1291.25 | 1292.31 | 1288.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 1285.75 | 1291.00 | 1288.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 15:00:00 | 1285.75 | 1291.00 | 1288.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 1290.00 | 1290.80 | 1288.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 09:15:00 | 1291.10 | 1290.80 | 1288.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 11:00:00 | 1291.30 | 1291.71 | 1289.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 10:15:00 | 1283.65 | 1293.73 | 1294.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 1283.65 | 1293.73 | 1294.23 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 09:15:00 | 1301.55 | 1293.90 | 1293.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 09:15:00 | 1319.80 | 1302.78 | 1300.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 09:15:00 | 1311.00 | 1315.17 | 1309.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-04 10:00:00 | 1311.00 | 1315.17 | 1309.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 1304.75 | 1313.09 | 1309.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:00:00 | 1304.75 | 1313.09 | 1309.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 11:15:00 | 1307.90 | 1312.05 | 1309.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:30:00 | 1308.65 | 1312.05 | 1309.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 12:15:00 | 1315.30 | 1312.70 | 1309.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 13:15:00 | 1318.05 | 1312.70 | 1309.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 14:15:00 | 1318.10 | 1313.07 | 1310.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 14:00:00 | 1316.00 | 1316.74 | 1314.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 09:15:00 | 1316.95 | 1313.93 | 1313.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 1312.00 | 1315.43 | 1314.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:00:00 | 1312.00 | 1315.43 | 1314.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 1309.80 | 1314.31 | 1314.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:45:00 | 1308.65 | 1314.31 | 1314.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-09-06 13:15:00 | 1307.75 | 1312.99 | 1313.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 13:15:00 | 1307.75 | 1312.99 | 1313.44 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 14:15:00 | 1320.40 | 1314.48 | 1314.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 11:15:00 | 1332.00 | 1319.08 | 1316.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 15:15:00 | 1337.45 | 1338.60 | 1332.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-11 09:15:00 | 1348.40 | 1338.60 | 1332.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 1337.70 | 1345.42 | 1340.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 1338.15 | 1345.42 | 1340.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 1335.00 | 1343.34 | 1339.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:45:00 | 1332.95 | 1343.34 | 1339.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 1335.40 | 1342.37 | 1340.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 10:00:00 | 1335.40 | 1342.37 | 1340.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 1339.05 | 1341.71 | 1340.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 11:45:00 | 1341.40 | 1342.34 | 1341.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-20 13:15:00 | 1348.10 | 1361.62 | 1362.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 13:15:00 | 1348.10 | 1361.62 | 1362.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 1336.35 | 1352.36 | 1357.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 15:15:00 | 1341.95 | 1341.67 | 1349.09 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:15:00 | 1328.65 | 1341.67 | 1349.09 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:45:00 | 1320.10 | 1338.52 | 1346.99 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 11:15:00 | 1329.25 | 1337.36 | 1345.69 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 12:45:00 | 1329.50 | 1334.31 | 1342.76 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 1293.65 | 1320.20 | 1332.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 14:00:00 | 1289.35 | 1304.73 | 1320.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 14:45:00 | 1290.55 | 1296.79 | 1307.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 09:15:00 | 1287.90 | 1295.83 | 1305.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 14:15:00 | 1301.90 | 1293.94 | 1299.86 | SL hit (close>ema400) qty=1.00 sl=1299.86 alert=retest1 |

### Cycle 34 — BUY (started 2023-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 14:15:00 | 1305.50 | 1300.41 | 1299.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 15:15:00 | 1310.00 | 1302.33 | 1300.67 | Break + close above crossover candle high |

### Cycle 35 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 1284.00 | 1298.66 | 1299.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 1275.40 | 1288.14 | 1292.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 10:15:00 | 1278.00 | 1276.83 | 1283.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-05 10:45:00 | 1277.35 | 1276.83 | 1283.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 1280.60 | 1277.05 | 1280.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 10:00:00 | 1280.60 | 1277.05 | 1280.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 1286.65 | 1278.97 | 1281.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 11:00:00 | 1286.65 | 1278.97 | 1281.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 11:15:00 | 1281.65 | 1279.51 | 1281.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 11:30:00 | 1282.25 | 1279.51 | 1281.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 12:15:00 | 1282.75 | 1280.16 | 1281.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:00:00 | 1282.75 | 1280.16 | 1281.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 1280.85 | 1280.30 | 1281.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:30:00 | 1280.00 | 1280.30 | 1281.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 1282.50 | 1280.74 | 1281.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 15:00:00 | 1282.50 | 1280.74 | 1281.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 1284.20 | 1281.43 | 1281.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 09:15:00 | 1278.00 | 1281.43 | 1281.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 11:15:00 | 1280.55 | 1281.56 | 1281.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 13:15:00 | 1283.75 | 1281.87 | 1281.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 13:15:00 | 1283.75 | 1281.87 | 1281.82 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 1276.05 | 1280.71 | 1281.29 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 15:15:00 | 1285.75 | 1281.72 | 1281.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 09:15:00 | 1287.45 | 1282.86 | 1282.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 09:15:00 | 1292.45 | 1295.68 | 1290.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 10:00:00 | 1292.45 | 1295.68 | 1290.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 1318.25 | 1317.02 | 1311.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:30:00 | 1313.40 | 1317.02 | 1311.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 1343.40 | 1349.79 | 1341.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:45:00 | 1342.35 | 1349.79 | 1341.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 1344.65 | 1348.76 | 1342.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 10:30:00 | 1343.65 | 1348.76 | 1342.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 1348.10 | 1347.80 | 1343.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:30:00 | 1342.80 | 1347.80 | 1343.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 1346.00 | 1347.79 | 1344.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:30:00 | 1341.15 | 1347.79 | 1344.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 10:15:00 | 1357.45 | 1357.79 | 1352.74 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 15:15:00 | 1335.05 | 1348.60 | 1350.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 11:15:00 | 1326.65 | 1342.01 | 1346.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 1317.00 | 1310.78 | 1320.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1317.00 | 1310.78 | 1320.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1317.00 | 1310.78 | 1320.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 1317.00 | 1310.78 | 1320.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 1311.50 | 1310.92 | 1319.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:30:00 | 1316.50 | 1310.92 | 1319.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 1308.70 | 1308.23 | 1314.27 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 14:15:00 | 1325.15 | 1316.78 | 1316.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 1360.30 | 1326.88 | 1321.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 10:15:00 | 1355.70 | 1355.97 | 1343.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 11:00:00 | 1355.70 | 1355.97 | 1343.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 1345.50 | 1352.89 | 1344.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 14:00:00 | 1345.50 | 1352.89 | 1344.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 14:15:00 | 1336.50 | 1349.61 | 1344.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 15:00:00 | 1336.50 | 1349.61 | 1344.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 15:15:00 | 1338.75 | 1347.44 | 1343.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 09:15:00 | 1355.05 | 1347.44 | 1343.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 13:00:00 | 1342.50 | 1345.99 | 1344.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 13:30:00 | 1342.35 | 1345.86 | 1344.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 09:15:00 | 1335.00 | 1342.70 | 1343.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 09:15:00 | 1335.00 | 1342.70 | 1343.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 14:15:00 | 1330.00 | 1336.40 | 1339.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 1336.50 | 1334.77 | 1338.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 1336.50 | 1334.77 | 1338.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 1336.50 | 1334.77 | 1338.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 10:30:00 | 1331.30 | 1333.82 | 1337.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 11:00:00 | 1330.00 | 1333.82 | 1337.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 15:15:00 | 1330.25 | 1331.11 | 1334.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 09:30:00 | 1330.65 | 1330.21 | 1333.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 1324.70 | 1329.10 | 1332.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 12:15:00 | 1321.25 | 1329.10 | 1332.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-07 13:15:00 | 1340.80 | 1331.10 | 1332.92 | SL hit (close>static) qty=1.00 sl=1340.70 alert=retest2 |

### Cycle 42 — BUY (started 2023-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 15:15:00 | 1343.25 | 1334.85 | 1334.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 12:15:00 | 1346.90 | 1339.25 | 1336.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 09:15:00 | 1342.75 | 1342.85 | 1339.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 09:15:00 | 1342.75 | 1342.85 | 1339.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 1342.75 | 1342.85 | 1339.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 11:15:00 | 1348.50 | 1343.47 | 1340.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 14:00:00 | 1347.65 | 1345.84 | 1342.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 15:00:00 | 1350.85 | 1346.84 | 1342.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 10:15:00 | 1335.15 | 1348.39 | 1347.71 | SL hit (close<static) qty=1.00 sl=1335.45 alert=retest2 |

### Cycle 43 — SELL (started 2023-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 11:15:00 | 1334.45 | 1345.60 | 1346.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 12:15:00 | 1333.00 | 1343.08 | 1345.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 1340.60 | 1336.95 | 1341.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 1340.60 | 1336.95 | 1341.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 1340.60 | 1336.95 | 1341.10 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-11-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 13:15:00 | 1354.80 | 1345.12 | 1344.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 14:15:00 | 1358.40 | 1347.78 | 1345.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 10:15:00 | 1392.85 | 1403.23 | 1387.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-20 11:00:00 | 1392.85 | 1403.23 | 1387.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 12:15:00 | 1388.15 | 1398.13 | 1387.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 12:30:00 | 1385.40 | 1398.13 | 1387.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 13:15:00 | 1385.80 | 1395.66 | 1387.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 13:45:00 | 1384.05 | 1395.66 | 1387.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 14:15:00 | 1384.40 | 1393.41 | 1387.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 15:00:00 | 1384.40 | 1393.41 | 1387.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 15:15:00 | 1387.00 | 1392.13 | 1387.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 09:15:00 | 1394.80 | 1392.13 | 1387.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 09:15:00 | 1405.50 | 1413.07 | 1413.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 09:15:00 | 1405.50 | 1413.07 | 1413.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 1394.00 | 1406.48 | 1409.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 13:15:00 | 1404.80 | 1402.38 | 1406.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-28 14:00:00 | 1404.80 | 1402.38 | 1406.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 1412.55 | 1404.41 | 1407.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 14:30:00 | 1411.50 | 1404.41 | 1407.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 1413.00 | 1406.13 | 1407.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:15:00 | 1416.70 | 1406.13 | 1407.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 1416.90 | 1409.71 | 1409.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 1434.80 | 1415.27 | 1412.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 09:15:00 | 1427.10 | 1430.68 | 1423.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 09:15:00 | 1427.10 | 1430.68 | 1423.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 1427.10 | 1430.68 | 1423.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 09:30:00 | 1421.55 | 1430.68 | 1423.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 1428.25 | 1430.19 | 1424.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 1434.40 | 1426.35 | 1424.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 09:15:00 | 1422.80 | 1425.64 | 1424.25 | SL hit (close<static) qty=1.00 sl=1424.00 alert=retest2 |

### Cycle 47 — SELL (started 2023-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 10:15:00 | 1454.40 | 1465.46 | 1466.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 14:15:00 | 1447.90 | 1453.59 | 1457.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 1403.70 | 1402.89 | 1415.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 15:00:00 | 1403.70 | 1402.89 | 1415.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1400.15 | 1402.86 | 1413.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 13:00:00 | 1398.00 | 1402.90 | 1410.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 10:15:00 | 1396.30 | 1400.53 | 1406.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-27 11:15:00 | 1421.60 | 1407.14 | 1405.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 11:15:00 | 1421.60 | 1407.14 | 1405.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 14:15:00 | 1422.25 | 1412.65 | 1408.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 1422.20 | 1424.82 | 1419.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 10:00:00 | 1422.20 | 1424.82 | 1419.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 1431.30 | 1429.14 | 1426.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 09:45:00 | 1424.70 | 1429.14 | 1426.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 1426.75 | 1428.67 | 1426.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:45:00 | 1426.55 | 1428.67 | 1426.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 1429.05 | 1428.74 | 1426.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:30:00 | 1427.45 | 1428.74 | 1426.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 1433.40 | 1433.29 | 1429.99 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 15:15:00 | 1420.80 | 1429.10 | 1429.43 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 10:15:00 | 1431.40 | 1429.70 | 1429.66 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-01-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 11:15:00 | 1427.60 | 1429.28 | 1429.47 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 1453.50 | 1434.17 | 1431.57 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 13:15:00 | 1427.95 | 1435.48 | 1436.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 1420.70 | 1432.52 | 1434.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 10:15:00 | 1430.50 | 1429.01 | 1432.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 10:15:00 | 1430.50 | 1429.01 | 1432.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 10:15:00 | 1430.50 | 1429.01 | 1432.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 11:00:00 | 1430.50 | 1429.01 | 1432.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 11:15:00 | 1435.35 | 1430.28 | 1432.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 12:00:00 | 1435.35 | 1430.28 | 1432.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 12:15:00 | 1457.50 | 1435.72 | 1434.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 13:15:00 | 1466.30 | 1441.84 | 1437.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 12:15:00 | 1457.75 | 1458.84 | 1449.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 13:15:00 | 1455.15 | 1458.10 | 1450.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 13:15:00 | 1455.15 | 1458.10 | 1450.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 13:45:00 | 1453.55 | 1458.10 | 1450.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 1450.65 | 1456.61 | 1450.15 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-01-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 12:15:00 | 1437.40 | 1446.55 | 1447.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-11 13:15:00 | 1427.60 | 1442.76 | 1445.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 14:15:00 | 1434.00 | 1430.89 | 1436.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 14:15:00 | 1434.00 | 1430.89 | 1436.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 14:15:00 | 1434.00 | 1430.89 | 1436.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 14:30:00 | 1436.00 | 1430.89 | 1436.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 15:15:00 | 1439.40 | 1432.59 | 1436.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 09:15:00 | 1434.35 | 1432.59 | 1436.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 1430.95 | 1432.26 | 1435.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 10:00:00 | 1421.35 | 1429.80 | 1432.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 15:00:00 | 1421.60 | 1423.68 | 1424.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-18 12:30:00 | 1421.75 | 1422.15 | 1423.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 10:15:00 | 1421.20 | 1417.36 | 1420.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 1418.30 | 1417.55 | 1420.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-19 13:15:00 | 1434.25 | 1423.79 | 1422.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 13:15:00 | 1434.25 | 1423.79 | 1422.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 1439.50 | 1426.93 | 1424.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 13:15:00 | 1434.95 | 1435.92 | 1430.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 14:00:00 | 1434.95 | 1435.92 | 1430.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 1442.50 | 1437.24 | 1431.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 09:15:00 | 1445.05 | 1438.58 | 1432.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 10:15:00 | 1414.25 | 1432.73 | 1431.22 | SL hit (close<static) qty=1.00 sl=1431.60 alert=retest2 |

### Cycle 57 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 1395.20 | 1425.22 | 1427.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 12:15:00 | 1375.55 | 1415.29 | 1423.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 1398.40 | 1396.08 | 1408.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 11:00:00 | 1398.40 | 1396.08 | 1408.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 1411.20 | 1397.57 | 1405.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:45:00 | 1408.45 | 1397.57 | 1405.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 1412.40 | 1400.53 | 1405.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 1393.35 | 1400.53 | 1405.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 1420.55 | 1391.35 | 1395.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 10:00:00 | 1420.55 | 1391.35 | 1395.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 10:15:00 | 1413.95 | 1395.87 | 1396.93 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 11:15:00 | 1409.30 | 1398.56 | 1398.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 14:15:00 | 1418.60 | 1405.66 | 1401.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 09:15:00 | 1407.15 | 1407.46 | 1403.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 09:15:00 | 1407.15 | 1407.46 | 1403.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 1407.15 | 1407.46 | 1403.29 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 09:15:00 | 1384.60 | 1401.01 | 1402.54 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 09:15:00 | 1413.10 | 1402.90 | 1402.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-01 11:15:00 | 1432.85 | 1409.68 | 1405.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 12:15:00 | 1422.15 | 1431.74 | 1422.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 12:15:00 | 1422.15 | 1431.74 | 1422.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 1422.15 | 1431.74 | 1422.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:00:00 | 1422.15 | 1431.74 | 1422.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 1426.20 | 1430.63 | 1422.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:45:00 | 1422.15 | 1430.63 | 1422.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 1427.30 | 1433.04 | 1426.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 10:30:00 | 1429.05 | 1433.04 | 1426.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 11:15:00 | 1427.65 | 1431.96 | 1426.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 12:15:00 | 1423.00 | 1431.96 | 1426.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 12:15:00 | 1427.30 | 1431.03 | 1426.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 12:30:00 | 1423.10 | 1431.03 | 1426.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 13:15:00 | 1420.25 | 1428.87 | 1426.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:00:00 | 1420.25 | 1428.87 | 1426.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 1419.55 | 1427.01 | 1425.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:45:00 | 1416.95 | 1427.01 | 1425.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 1436.45 | 1427.13 | 1425.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 14:00:00 | 1441.40 | 1432.16 | 1428.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 13:15:00 | 1446.75 | 1456.67 | 1456.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 13:15:00 | 1446.75 | 1456.67 | 1456.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 14:15:00 | 1443.15 | 1453.97 | 1455.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 1451.30 | 1440.36 | 1445.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 09:15:00 | 1451.30 | 1440.36 | 1445.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 1451.30 | 1440.36 | 1445.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:00:00 | 1451.30 | 1440.36 | 1445.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 1449.85 | 1442.26 | 1445.46 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 12:15:00 | 1465.75 | 1450.11 | 1448.66 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 13:15:00 | 1443.65 | 1450.39 | 1450.82 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 1458.00 | 1452.04 | 1451.51 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 10:15:00 | 1447.25 | 1450.58 | 1450.90 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 15:15:00 | 1458.90 | 1452.36 | 1451.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 1470.55 | 1456.00 | 1453.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 1487.15 | 1489.38 | 1474.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 10:00:00 | 1487.15 | 1489.38 | 1474.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 1492.25 | 1490.40 | 1482.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 10:45:00 | 1498.45 | 1492.80 | 1484.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 11:15:00 | 1480.45 | 1493.96 | 1495.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 11:15:00 | 1480.45 | 1493.96 | 1495.62 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 1511.00 | 1498.01 | 1496.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 10:15:00 | 1514.40 | 1502.69 | 1499.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 14:15:00 | 1543.05 | 1549.38 | 1538.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-27 15:00:00 | 1543.05 | 1549.38 | 1538.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 1542.45 | 1548.00 | 1538.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:15:00 | 1539.70 | 1548.00 | 1538.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 1548.65 | 1548.13 | 1539.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 12:15:00 | 1555.40 | 1546.76 | 1540.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 13:30:00 | 1553.55 | 1548.72 | 1542.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 13:45:00 | 1553.25 | 1545.76 | 1543.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 15:00:00 | 1553.45 | 1547.30 | 1544.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 1549.75 | 1551.03 | 1547.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-01 12:00:00 | 1549.75 | 1551.03 | 1547.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 13:15:00 | 1547.45 | 1550.05 | 1547.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-01 13:30:00 | 1548.75 | 1550.05 | 1547.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 14:15:00 | 1540.80 | 1548.20 | 1547.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-01 15:00:00 | 1540.80 | 1548.20 | 1547.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 1541.00 | 1546.76 | 1546.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-02 09:15:00 | 1545.60 | 1546.76 | 1546.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-04 09:15:00 | 1532.35 | 1544.24 | 1545.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 09:15:00 | 1532.35 | 1544.24 | 1545.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 10:15:00 | 1524.05 | 1540.20 | 1543.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 1504.10 | 1498.34 | 1508.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 14:00:00 | 1504.10 | 1498.34 | 1508.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 1520.15 | 1502.70 | 1509.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 14:30:00 | 1518.80 | 1502.70 | 1509.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 1523.65 | 1506.89 | 1510.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 09:15:00 | 1517.60 | 1506.89 | 1510.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-11 09:15:00 | 1525.00 | 1508.63 | 1508.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 09:15:00 | 1525.00 | 1508.63 | 1508.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-12 09:15:00 | 1544.90 | 1527.58 | 1519.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 11:15:00 | 1522.25 | 1526.92 | 1520.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 11:15:00 | 1522.25 | 1526.92 | 1520.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 1522.25 | 1526.92 | 1520.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:00:00 | 1522.25 | 1526.92 | 1520.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 12:15:00 | 1530.00 | 1527.54 | 1521.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:30:00 | 1522.50 | 1527.54 | 1521.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 13:15:00 | 1519.45 | 1525.92 | 1521.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 14:00:00 | 1519.45 | 1525.92 | 1521.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 1516.20 | 1523.98 | 1520.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 15:00:00 | 1516.20 | 1523.98 | 1520.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 15:15:00 | 1515.00 | 1522.18 | 1520.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-13 09:15:00 | 1521.65 | 1522.18 | 1520.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-13 09:15:00 | 1511.10 | 1519.97 | 1519.38 | SL hit (close<static) qty=1.00 sl=1512.50 alert=retest2 |

### Cycle 71 — SELL (started 2024-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 10:15:00 | 1503.15 | 1516.60 | 1517.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 1488.55 | 1510.99 | 1515.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 1504.10 | 1500.24 | 1506.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 10:15:00 | 1504.10 | 1500.24 | 1506.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 1504.10 | 1500.24 | 1506.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:30:00 | 1501.10 | 1500.24 | 1506.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 1505.10 | 1501.21 | 1506.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:45:00 | 1512.20 | 1501.21 | 1506.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 1497.95 | 1500.56 | 1505.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 13:45:00 | 1493.40 | 1499.25 | 1504.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-14 15:15:00 | 1507.85 | 1501.67 | 1504.96 | SL hit (close>static) qty=1.00 sl=1505.90 alert=retest2 |

### Cycle 72 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 1492.60 | 1474.23 | 1471.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 13:15:00 | 1499.15 | 1479.21 | 1474.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 12:15:00 | 1485.00 | 1486.94 | 1481.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 12:30:00 | 1486.85 | 1486.94 | 1481.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 1496.75 | 1489.50 | 1484.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 10:00:00 | 1500.00 | 1488.67 | 1486.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 10:30:00 | 1500.00 | 1490.36 | 1487.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 12:15:00 | 1499.55 | 1491.81 | 1488.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 15:15:00 | 1507.30 | 1498.49 | 1492.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 1507.30 | 1500.25 | 1494.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-02 09:15:00 | 1471.65 | 1490.24 | 1492.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 09:15:00 | 1471.65 | 1490.24 | 1492.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-02 14:15:00 | 1470.20 | 1478.65 | 1485.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 10:15:00 | 1463.50 | 1463.35 | 1470.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-04 10:45:00 | 1456.75 | 1463.35 | 1470.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 1472.25 | 1464.77 | 1470.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 13:00:00 | 1472.25 | 1464.77 | 1470.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 1464.50 | 1464.71 | 1469.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 13:30:00 | 1470.05 | 1464.71 | 1469.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 15:15:00 | 1465.00 | 1464.47 | 1468.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:15:00 | 1479.00 | 1464.47 | 1468.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 1481.20 | 1467.82 | 1469.89 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-04-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 11:15:00 | 1476.50 | 1471.66 | 1471.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 12:15:00 | 1482.50 | 1473.83 | 1472.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 13:15:00 | 1511.45 | 1514.39 | 1503.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-09 13:45:00 | 1508.65 | 1514.39 | 1503.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 1504.70 | 1511.65 | 1504.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:30:00 | 1501.00 | 1508.72 | 1503.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 1489.10 | 1504.80 | 1502.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:30:00 | 1487.85 | 1504.80 | 1502.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 1491.25 | 1502.09 | 1501.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 11:45:00 | 1488.45 | 1502.09 | 1501.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 15:15:00 | 1495.00 | 1501.19 | 1501.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 1499.95 | 1501.19 | 1501.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 1508.25 | 1502.60 | 1501.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 10:30:00 | 1510.60 | 1502.22 | 1501.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 11:15:00 | 1495.05 | 1500.79 | 1501.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 11:15:00 | 1495.05 | 1500.79 | 1501.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 1493.85 | 1497.96 | 1499.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 15:15:00 | 1467.90 | 1466.64 | 1474.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 09:15:00 | 1468.75 | 1466.64 | 1474.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 1465.85 | 1466.49 | 1473.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 15:00:00 | 1453.40 | 1467.71 | 1472.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 09:15:00 | 1428.55 | 1466.12 | 1471.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 10:15:00 | 1478.55 | 1459.90 | 1461.88 | SL hit (close>static) qty=1.00 sl=1476.60 alert=retest2 |

### Cycle 76 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 11:15:00 | 1478.25 | 1463.57 | 1463.37 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 12:15:00 | 1456.55 | 1468.86 | 1469.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-25 09:15:00 | 1444.00 | 1460.64 | 1465.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 09:15:00 | 1437.40 | 1430.73 | 1441.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 09:15:00 | 1437.40 | 1430.73 | 1441.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 1437.40 | 1430.73 | 1441.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-29 12:00:00 | 1421.50 | 1429.43 | 1438.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 1447.00 | 1436.25 | 1435.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 09:15:00 | 1447.00 | 1436.25 | 1435.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 11:15:00 | 1458.20 | 1442.95 | 1438.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 11:15:00 | 1453.75 | 1456.43 | 1449.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-03 11:30:00 | 1460.00 | 1456.43 | 1449.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 1452.00 | 1455.54 | 1449.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:45:00 | 1445.70 | 1455.54 | 1449.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 1450.00 | 1454.43 | 1449.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:30:00 | 1447.95 | 1454.43 | 1449.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 1443.00 | 1452.15 | 1448.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 15:00:00 | 1443.00 | 1452.15 | 1448.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 1442.00 | 1450.12 | 1448.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 09:15:00 | 1453.95 | 1450.12 | 1448.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 09:15:00 | 1437.55 | 1447.60 | 1447.35 | SL hit (close<static) qty=1.00 sl=1438.40 alert=retest2 |

### Cycle 79 — SELL (started 2024-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 10:15:00 | 1441.10 | 1446.30 | 1446.78 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 12:15:00 | 1448.00 | 1445.53 | 1445.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 09:15:00 | 1463.50 | 1450.90 | 1447.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 10:15:00 | 1445.45 | 1449.81 | 1447.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 10:15:00 | 1445.45 | 1449.81 | 1447.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 1445.45 | 1449.81 | 1447.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 1445.45 | 1449.81 | 1447.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 1445.85 | 1449.02 | 1447.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:00:00 | 1445.85 | 1449.02 | 1447.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 12:15:00 | 1436.35 | 1446.48 | 1446.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 13:15:00 | 1430.00 | 1443.19 | 1445.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 1430.90 | 1426.90 | 1432.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 1430.90 | 1426.90 | 1432.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 1430.90 | 1426.90 | 1432.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:30:00 | 1428.05 | 1426.90 | 1432.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 1430.70 | 1427.66 | 1432.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 14:45:00 | 1424.10 | 1428.04 | 1429.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 12:15:00 | 1424.45 | 1427.18 | 1428.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 15:15:00 | 1436.50 | 1430.20 | 1429.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 15:15:00 | 1436.50 | 1430.20 | 1429.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 1440.00 | 1432.74 | 1431.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 1430.10 | 1432.60 | 1431.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 12:15:00 | 1430.10 | 1432.60 | 1431.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 1430.10 | 1432.60 | 1431.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:45:00 | 1429.70 | 1432.60 | 1431.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 1434.90 | 1433.06 | 1431.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:45:00 | 1433.10 | 1433.06 | 1431.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 1429.20 | 1432.29 | 1431.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 1429.20 | 1432.29 | 1431.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 1428.80 | 1431.59 | 1431.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:15:00 | 1426.00 | 1431.59 | 1431.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 09:15:00 | 1426.00 | 1430.47 | 1430.72 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 11:15:00 | 1440.60 | 1431.46 | 1431.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 14:15:00 | 1453.10 | 1436.66 | 1433.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 1437.25 | 1438.95 | 1435.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 09:15:00 | 1437.25 | 1438.95 | 1435.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 1437.25 | 1438.95 | 1435.32 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 1428.35 | 1435.34 | 1435.73 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 1438.75 | 1432.23 | 1431.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 12:15:00 | 1446.40 | 1436.82 | 1434.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 1437.90 | 1440.59 | 1437.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 1437.90 | 1440.59 | 1437.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1437.90 | 1440.59 | 1437.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 1439.05 | 1440.59 | 1437.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1444.75 | 1441.42 | 1437.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:30:00 | 1437.90 | 1441.42 | 1437.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 1442.10 | 1441.56 | 1438.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:00:00 | 1442.10 | 1441.56 | 1438.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 1439.10 | 1441.39 | 1438.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:00:00 | 1439.10 | 1441.39 | 1438.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 1436.90 | 1440.49 | 1438.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 1436.90 | 1440.49 | 1438.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1437.00 | 1439.79 | 1438.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 1437.25 | 1439.79 | 1438.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1436.00 | 1439.04 | 1438.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:30:00 | 1438.10 | 1439.04 | 1438.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 1434.25 | 1438.08 | 1437.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 1439.85 | 1438.75 | 1438.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 1431.00 | 1437.05 | 1437.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 13:15:00 | 1431.00 | 1437.05 | 1437.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 1409.20 | 1431.48 | 1434.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 1426.70 | 1426.61 | 1431.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 10:00:00 | 1426.70 | 1426.61 | 1431.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1430.20 | 1427.33 | 1431.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:45:00 | 1429.45 | 1427.33 | 1431.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 1432.45 | 1428.35 | 1431.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 1432.45 | 1428.35 | 1431.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 1434.10 | 1429.50 | 1432.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:45:00 | 1434.90 | 1429.50 | 1432.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 1434.00 | 1430.40 | 1432.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:45:00 | 1437.90 | 1430.40 | 1432.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 14:15:00 | 1450.20 | 1434.36 | 1433.83 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 1417.05 | 1431.05 | 1432.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 11:15:00 | 1408.15 | 1426.47 | 1430.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 1395.30 | 1388.73 | 1399.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 12:45:00 | 1395.05 | 1388.73 | 1399.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 1398.05 | 1390.59 | 1399.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:45:00 | 1400.00 | 1390.59 | 1399.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1382.60 | 1388.99 | 1397.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1355.05 | 1393.02 | 1396.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 1401.30 | 1394.68 | 1396.50 | SL hit (close>static) qty=1.00 sl=1398.75 alert=retest2 |

### Cycle 90 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 1385.00 | 1378.37 | 1378.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1421.10 | 1386.91 | 1381.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 13:15:00 | 1421.05 | 1422.06 | 1410.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 13:30:00 | 1420.70 | 1422.06 | 1410.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1436.50 | 1432.62 | 1424.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 10:30:00 | 1442.35 | 1433.81 | 1428.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:15:00 | 1440.35 | 1445.48 | 1440.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 14:15:00 | 1448.70 | 1458.23 | 1458.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 1448.70 | 1458.23 | 1458.32 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 1465.95 | 1457.81 | 1457.36 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 11:15:00 | 1456.95 | 1459.13 | 1459.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 12:15:00 | 1451.05 | 1457.51 | 1458.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 12:15:00 | 1453.80 | 1448.19 | 1452.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 12:15:00 | 1453.80 | 1448.19 | 1452.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 1453.80 | 1448.19 | 1452.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:45:00 | 1454.95 | 1448.19 | 1452.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 1455.75 | 1449.70 | 1452.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:00:00 | 1455.75 | 1449.70 | 1452.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 1464.30 | 1452.62 | 1453.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 1464.30 | 1452.62 | 1453.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 1461.00 | 1454.30 | 1454.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 1457.60 | 1454.30 | 1454.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 1465.70 | 1456.58 | 1455.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 1465.70 | 1456.58 | 1455.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 1467.55 | 1458.77 | 1456.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 11:15:00 | 1457.80 | 1458.58 | 1456.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 12:00:00 | 1457.80 | 1458.58 | 1456.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 1450.25 | 1456.91 | 1456.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:00:00 | 1450.25 | 1456.91 | 1456.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 1453.10 | 1456.15 | 1455.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:45:00 | 1450.40 | 1456.15 | 1455.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 1451.60 | 1455.24 | 1455.48 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 1461.35 | 1455.75 | 1455.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 1491.75 | 1465.72 | 1460.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 1473.95 | 1492.34 | 1484.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 1473.95 | 1492.34 | 1484.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 1473.95 | 1492.34 | 1484.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 1469.80 | 1492.34 | 1484.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1490.65 | 1492.01 | 1485.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 11:45:00 | 1496.55 | 1492.15 | 1486.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:00:00 | 1496.95 | 1492.35 | 1487.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 1498.60 | 1492.56 | 1488.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-18 09:15:00 | 1646.21 | 1614.83 | 1596.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 15:15:00 | 1619.10 | 1628.86 | 1629.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 1610.35 | 1625.16 | 1627.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 1625.15 | 1609.61 | 1615.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 1625.15 | 1609.61 | 1615.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1625.15 | 1609.61 | 1615.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1625.15 | 1609.61 | 1615.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1643.65 | 1616.42 | 1618.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 1649.70 | 1616.42 | 1618.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 1610.00 | 1616.49 | 1618.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 14:15:00 | 1625.00 | 1616.49 | 1618.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 1635.40 | 1620.27 | 1619.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 1662.55 | 1630.71 | 1624.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 1710.90 | 1737.64 | 1721.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 1710.90 | 1737.64 | 1721.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1710.90 | 1737.64 | 1721.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 1710.90 | 1737.64 | 1721.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1725.00 | 1735.11 | 1721.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:15:00 | 1727.00 | 1735.11 | 1721.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:45:00 | 1733.25 | 1732.01 | 1725.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 1737.90 | 1750.03 | 1750.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 1737.90 | 1750.03 | 1750.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 1674.30 | 1698.31 | 1716.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-08 11:15:00 | 1691.60 | 1685.09 | 1695.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-08 12:00:00 | 1691.60 | 1685.09 | 1695.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 1683.75 | 1684.82 | 1694.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:15:00 | 1679.25 | 1684.82 | 1694.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 14:15:00 | 1705.30 | 1690.63 | 1695.23 | SL hit (close>static) qty=1.00 sl=1698.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 1728.10 | 1703.35 | 1700.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 1730.05 | 1715.97 | 1707.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 1700.70 | 1716.16 | 1709.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 1700.70 | 1716.16 | 1709.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1700.70 | 1716.16 | 1709.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 1700.20 | 1716.16 | 1709.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 1717.00 | 1716.32 | 1710.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 11:45:00 | 1727.80 | 1717.11 | 1711.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 09:15:00 | 1681.10 | 1704.72 | 1707.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 1681.10 | 1704.72 | 1707.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1676.45 | 1692.20 | 1700.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 10:15:00 | 1694.40 | 1689.23 | 1695.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 10:15:00 | 1694.40 | 1689.23 | 1695.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1694.40 | 1689.23 | 1695.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 1694.40 | 1689.23 | 1695.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 1687.20 | 1688.83 | 1694.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:30:00 | 1693.90 | 1688.83 | 1694.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 1700.45 | 1691.15 | 1695.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:00:00 | 1700.45 | 1691.15 | 1695.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 1696.95 | 1692.31 | 1695.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 15:15:00 | 1688.75 | 1691.90 | 1695.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 09:30:00 | 1680.65 | 1687.81 | 1692.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 14:30:00 | 1687.25 | 1687.60 | 1690.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 15:00:00 | 1687.65 | 1687.60 | 1690.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1690.90 | 1688.26 | 1690.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 1684.10 | 1688.26 | 1690.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1681.20 | 1686.85 | 1689.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 1676.50 | 1686.85 | 1689.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 09:15:00 | 1705.85 | 1680.92 | 1683.37 | SL hit (close>static) qty=1.00 sl=1699.90 alert=retest2 |

### Cycle 102 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 1716.80 | 1688.09 | 1686.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 11:15:00 | 1719.90 | 1694.45 | 1689.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 1794.75 | 1795.22 | 1772.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 15:00:00 | 1794.75 | 1795.22 | 1772.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1796.60 | 1793.52 | 1784.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 13:15:00 | 1799.55 | 1793.70 | 1786.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:45:00 | 1804.45 | 1796.93 | 1790.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:45:00 | 1806.10 | 1799.83 | 1792.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 1893.45 | 1900.57 | 1900.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 1893.45 | 1900.57 | 1900.86 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 09:15:00 | 1910.50 | 1902.94 | 1901.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 11:15:00 | 1928.95 | 1911.54 | 1906.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 13:15:00 | 1898.35 | 1911.05 | 1907.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 13:15:00 | 1898.35 | 1911.05 | 1907.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1898.35 | 1911.05 | 1907.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:45:00 | 1896.20 | 1911.05 | 1907.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1901.95 | 1909.23 | 1906.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 1898.10 | 1909.23 | 1906.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 09:15:00 | 1843.00 | 1894.03 | 1900.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 09:15:00 | 1832.00 | 1848.00 | 1856.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 10:15:00 | 1831.90 | 1828.50 | 1838.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 11:00:00 | 1831.90 | 1828.50 | 1838.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1840.25 | 1826.57 | 1832.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:45:00 | 1841.10 | 1826.57 | 1832.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1847.25 | 1830.71 | 1834.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 1847.25 | 1830.71 | 1834.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 1842.15 | 1836.34 | 1836.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 09:15:00 | 1854.55 | 1841.05 | 1838.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 11:15:00 | 1838.90 | 1841.52 | 1839.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 11:15:00 | 1838.90 | 1841.52 | 1839.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1838.90 | 1841.52 | 1839.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 12:00:00 | 1838.90 | 1841.52 | 1839.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 1829.70 | 1839.16 | 1838.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 1829.70 | 1839.16 | 1838.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 1835.40 | 1838.41 | 1837.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 14:45:00 | 1840.50 | 1839.53 | 1838.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 1859.05 | 1876.77 | 1877.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 1859.05 | 1876.77 | 1877.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 10:15:00 | 1845.60 | 1870.54 | 1874.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 1865.05 | 1863.52 | 1869.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 15:00:00 | 1865.05 | 1863.52 | 1869.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1871.00 | 1865.02 | 1869.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 1888.85 | 1865.02 | 1869.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1887.05 | 1869.43 | 1871.30 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 1889.20 | 1873.38 | 1872.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 1898.00 | 1882.72 | 1877.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 14:15:00 | 1883.90 | 1895.62 | 1888.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 1883.90 | 1895.62 | 1888.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1883.90 | 1895.62 | 1888.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 1883.90 | 1895.62 | 1888.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 1893.00 | 1895.09 | 1889.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 1890.95 | 1895.09 | 1889.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1881.75 | 1892.43 | 1888.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:00:00 | 1881.75 | 1892.43 | 1888.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1870.10 | 1887.96 | 1886.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 1870.10 | 1887.96 | 1886.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 1860.30 | 1882.43 | 1884.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1848.70 | 1872.18 | 1879.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 1828.30 | 1817.79 | 1832.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 1828.30 | 1817.79 | 1832.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1828.30 | 1817.79 | 1832.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:45:00 | 1840.65 | 1817.79 | 1832.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1797.85 | 1805.05 | 1818.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:45:00 | 1787.40 | 1802.05 | 1815.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:45:00 | 1788.35 | 1799.61 | 1813.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 13:15:00 | 1746.40 | 1740.45 | 1739.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 1746.40 | 1740.45 | 1739.76 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 14:15:00 | 1722.15 | 1736.79 | 1738.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 1707.85 | 1727.93 | 1732.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 1707.45 | 1706.62 | 1714.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:45:00 | 1708.85 | 1706.62 | 1714.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1713.70 | 1707.46 | 1712.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:30:00 | 1710.00 | 1708.96 | 1713.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 1710.00 | 1709.17 | 1712.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 1709.50 | 1707.69 | 1711.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:15:00 | 1707.00 | 1708.75 | 1711.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 1707.00 | 1708.40 | 1711.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:30:00 | 1719.95 | 1711.11 | 1712.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1713.80 | 1711.65 | 1712.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 1720.85 | 1711.65 | 1712.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 1696.35 | 1708.59 | 1710.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-23 12:15:00 | 1718.95 | 1709.68 | 1709.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 1718.95 | 1709.68 | 1709.28 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 09:15:00 | 1641.85 | 1698.09 | 1704.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 1616.80 | 1641.50 | 1665.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 15:15:00 | 1615.00 | 1610.55 | 1626.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 09:15:00 | 1610.70 | 1610.55 | 1626.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 1625.70 | 1613.01 | 1624.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:00:00 | 1625.70 | 1613.01 | 1624.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 1638.50 | 1618.11 | 1626.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:00:00 | 1638.50 | 1618.11 | 1626.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 1657.00 | 1625.89 | 1628.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:45:00 | 1658.80 | 1625.89 | 1628.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 1660.50 | 1632.81 | 1631.82 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 1625.35 | 1633.29 | 1634.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 1616.50 | 1625.58 | 1629.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 1626.05 | 1623.94 | 1627.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 1626.05 | 1623.94 | 1627.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1626.05 | 1623.94 | 1627.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1626.05 | 1623.94 | 1627.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1633.65 | 1625.88 | 1628.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 1633.65 | 1625.88 | 1628.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1604.30 | 1621.56 | 1625.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:30:00 | 1601.45 | 1615.93 | 1622.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 11:30:00 | 1601.85 | 1613.98 | 1621.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 12:15:00 | 1600.55 | 1613.98 | 1621.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 12:45:00 | 1600.25 | 1611.93 | 1619.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1609.60 | 1610.03 | 1616.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:30:00 | 1608.95 | 1610.03 | 1616.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 1612.50 | 1610.52 | 1615.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:30:00 | 1602.00 | 1609.08 | 1614.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 12:15:00 | 1601.40 | 1609.08 | 1614.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 13:15:00 | 1639.30 | 1615.17 | 1616.57 | SL hit (close>static) qty=1.00 sl=1619.05 alert=retest2 |

### Cycle 116 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 1631.00 | 1618.34 | 1617.88 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 10:15:00 | 1614.40 | 1617.47 | 1617.66 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 1618.55 | 1617.82 | 1617.79 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 13:15:00 | 1614.35 | 1617.13 | 1617.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-06 14:15:00 | 1602.00 | 1614.10 | 1616.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 1599.00 | 1593.14 | 1600.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 1599.00 | 1593.14 | 1600.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1599.00 | 1593.14 | 1600.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 1594.60 | 1593.14 | 1600.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 1590.35 | 1592.58 | 1599.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 1584.15 | 1592.58 | 1599.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 12:15:00 | 1504.94 | 1539.95 | 1551.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 11:15:00 | 1490.20 | 1489.61 | 1507.10 | SL hit (close>ema200) qty=0.50 sl=1489.61 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 11:15:00 | 1514.95 | 1501.58 | 1501.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 1515.40 | 1506.98 | 1504.98 | Break + close above crossover candle high |

### Cycle 121 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 1448.15 | 1496.42 | 1500.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 1415.80 | 1480.30 | 1493.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 14:15:00 | 1438.45 | 1426.23 | 1451.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 15:00:00 | 1438.45 | 1426.23 | 1451.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1432.75 | 1429.10 | 1448.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 1440.55 | 1429.10 | 1448.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1422.95 | 1424.09 | 1436.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 10:30:00 | 1419.10 | 1423.55 | 1434.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 13:15:00 | 1444.55 | 1429.68 | 1434.85 | SL hit (close>static) qty=1.00 sl=1438.45 alert=retest2 |

### Cycle 122 — BUY (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 09:15:00 | 1463.10 | 1439.21 | 1438.21 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 11:15:00 | 1430.00 | 1440.94 | 1442.37 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 14:15:00 | 1449.20 | 1440.89 | 1440.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 1462.95 | 1446.17 | 1442.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 11:15:00 | 1467.95 | 1469.65 | 1460.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 12:00:00 | 1467.95 | 1469.65 | 1460.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 1459.55 | 1467.26 | 1460.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:00:00 | 1459.55 | 1467.26 | 1460.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 1462.50 | 1466.31 | 1461.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:45:00 | 1456.50 | 1466.31 | 1461.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 1462.00 | 1465.44 | 1461.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 1461.30 | 1465.44 | 1461.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1458.85 | 1464.13 | 1460.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 10:30:00 | 1467.50 | 1464.16 | 1461.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 11:30:00 | 1464.20 | 1464.33 | 1461.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 1438.20 | 1457.17 | 1459.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 1438.20 | 1457.17 | 1459.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 1434.50 | 1452.64 | 1457.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 1428.70 | 1428.45 | 1437.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 1426.85 | 1428.28 | 1436.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1426.85 | 1428.28 | 1436.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 1420.75 | 1428.28 | 1436.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:00:00 | 1421.65 | 1425.22 | 1433.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 1421.95 | 1424.62 | 1430.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:45:00 | 1420.75 | 1422.62 | 1428.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1407.10 | 1412.99 | 1420.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:30:00 | 1419.60 | 1412.99 | 1420.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 1402.90 | 1398.90 | 1406.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:45:00 | 1403.30 | 1398.90 | 1406.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 1406.45 | 1400.41 | 1406.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 1406.45 | 1400.41 | 1406.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1406.45 | 1401.61 | 1406.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 15:15:00 | 1402.30 | 1401.61 | 1406.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 1398.75 | 1400.04 | 1404.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:45:00 | 1403.40 | 1403.31 | 1405.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 1394.30 | 1402.22 | 1404.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1403.05 | 1401.98 | 1403.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 1403.05 | 1401.98 | 1403.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 1401.35 | 1401.85 | 1403.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 13:00:00 | 1396.55 | 1400.79 | 1402.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 1407.90 | 1403.77 | 1403.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 1407.90 | 1403.77 | 1403.73 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 10:15:00 | 1403.15 | 1403.65 | 1403.68 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 1406.40 | 1404.20 | 1403.93 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 14:15:00 | 1386.00 | 1401.20 | 1402.70 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 1408.50 | 1403.29 | 1403.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 1409.50 | 1405.30 | 1404.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 13:15:00 | 1406.00 | 1408.12 | 1406.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 13:15:00 | 1406.00 | 1408.12 | 1406.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1406.00 | 1408.12 | 1406.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:45:00 | 1407.15 | 1408.12 | 1406.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 1404.85 | 1407.46 | 1406.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:45:00 | 1404.55 | 1407.46 | 1406.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 1400.80 | 1406.13 | 1405.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 1402.35 | 1406.13 | 1405.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1404.55 | 1405.81 | 1405.66 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 1403.70 | 1405.54 | 1405.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 1396.20 | 1403.67 | 1404.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 1404.40 | 1402.11 | 1403.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 1404.40 | 1402.11 | 1403.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1404.40 | 1402.11 | 1403.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 1404.40 | 1402.11 | 1403.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1401.95 | 1402.08 | 1403.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 1387.20 | 1402.08 | 1403.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 1390.00 | 1395.09 | 1399.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 13:30:00 | 1390.10 | 1394.27 | 1398.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 1390.00 | 1394.27 | 1398.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1390.05 | 1393.43 | 1397.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 1397.80 | 1393.43 | 1397.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1391.55 | 1392.66 | 1396.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 1402.00 | 1398.35 | 1397.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 1402.00 | 1398.35 | 1397.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 1423.20 | 1409.16 | 1403.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 1437.35 | 1439.83 | 1427.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 1437.35 | 1439.83 | 1427.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1435.55 | 1438.97 | 1428.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1427.60 | 1438.97 | 1428.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 1429.95 | 1436.01 | 1429.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:45:00 | 1433.80 | 1436.01 | 1429.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 1437.50 | 1436.31 | 1430.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 1446.65 | 1436.31 | 1430.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 13:15:00 | 1497.70 | 1508.33 | 1508.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 13:15:00 | 1497.70 | 1508.33 | 1508.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 1482.15 | 1500.24 | 1504.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 15:15:00 | 1464.95 | 1464.46 | 1475.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 09:15:00 | 1450.50 | 1464.46 | 1475.89 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 11:15:00 | 1457.00 | 1462.50 | 1472.95 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 12:15:00 | 1457.10 | 1462.00 | 1471.77 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 14:15:00 | 1456.85 | 1460.83 | 1469.50 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1424.30 | 1427.40 | 1435.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:00:00 | 1420.20 | 1425.96 | 1433.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 1438.25 | 1427.23 | 1432.90 | SL hit (close>ema400) qty=1.00 sl=1432.90 alert=retest1 |

### Cycle 134 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 1449.95 | 1437.26 | 1436.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 1457.55 | 1441.31 | 1438.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 10:15:00 | 1466.65 | 1477.91 | 1469.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 10:15:00 | 1466.65 | 1477.91 | 1469.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1466.65 | 1477.91 | 1469.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 1466.65 | 1477.91 | 1469.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1480.45 | 1478.42 | 1470.69 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1395.05 | 1461.74 | 1463.82 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 1465.45 | 1460.23 | 1459.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 13:15:00 | 1472.10 | 1463.19 | 1461.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 1470.50 | 1472.13 | 1468.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 14:15:00 | 1470.50 | 1472.13 | 1468.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 1470.50 | 1472.13 | 1468.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 15:00:00 | 1470.50 | 1472.13 | 1468.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 1466.05 | 1470.91 | 1467.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 1461.85 | 1470.91 | 1467.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1465.15 | 1469.76 | 1467.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 1461.45 | 1469.76 | 1467.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1469.35 | 1469.68 | 1467.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:30:00 | 1466.15 | 1469.68 | 1467.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1476.30 | 1471.00 | 1468.57 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 1465.00 | 1468.20 | 1468.35 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 14:15:00 | 1471.85 | 1468.93 | 1468.67 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1465.00 | 1468.32 | 1468.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 1453.00 | 1465.25 | 1467.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1446.05 | 1426.77 | 1437.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 1446.05 | 1426.77 | 1437.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1446.05 | 1426.77 | 1437.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 1446.05 | 1426.77 | 1437.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1458.25 | 1433.06 | 1439.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 1456.65 | 1433.06 | 1439.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 13:15:00 | 1457.55 | 1443.05 | 1442.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 10:15:00 | 1471.80 | 1453.89 | 1448.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 09:15:00 | 1463.65 | 1464.96 | 1457.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-14 10:00:00 | 1463.65 | 1464.96 | 1457.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 1458.80 | 1463.73 | 1457.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 1458.80 | 1463.73 | 1457.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 1456.50 | 1462.28 | 1457.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:00:00 | 1456.50 | 1462.28 | 1457.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 1452.00 | 1460.23 | 1457.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:30:00 | 1454.70 | 1460.23 | 1457.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 1472.50 | 1463.07 | 1459.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 10:30:00 | 1479.60 | 1465.09 | 1460.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 11:30:00 | 1479.80 | 1467.49 | 1462.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 13:15:00 | 1478.85 | 1468.99 | 1463.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-18 13:30:00 | 1479.55 | 1472.07 | 1468.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1474.30 | 1473.41 | 1469.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 1483.00 | 1476.18 | 1472.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 10:15:00 | 1482.55 | 1477.00 | 1473.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 15:15:00 | 1470.00 | 1471.90 | 1472.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 15:15:00 | 1470.00 | 1471.90 | 1472.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 1466.65 | 1470.85 | 1471.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 10:15:00 | 1470.95 | 1470.87 | 1471.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 1470.95 | 1470.87 | 1471.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1470.95 | 1470.87 | 1471.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:30:00 | 1473.00 | 1470.87 | 1471.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 11:15:00 | 1476.50 | 1471.99 | 1471.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 12:15:00 | 1480.45 | 1473.69 | 1472.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 1480.55 | 1480.80 | 1476.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 1482.30 | 1480.80 | 1476.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1488.40 | 1482.32 | 1477.74 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 1467.00 | 1477.18 | 1478.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 1462.05 | 1468.24 | 1471.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 1444.50 | 1439.24 | 1450.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 09:30:00 | 1446.60 | 1439.24 | 1450.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 1440.00 | 1439.39 | 1449.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 10:30:00 | 1445.90 | 1439.39 | 1449.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 1421.30 | 1407.40 | 1414.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 14:00:00 | 1421.30 | 1407.40 | 1414.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 1419.85 | 1409.89 | 1415.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 15:00:00 | 1419.85 | 1409.89 | 1415.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 1416.60 | 1411.23 | 1415.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:15:00 | 1402.00 | 1411.23 | 1415.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 1406.85 | 1410.36 | 1414.70 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 14:15:00 | 1421.60 | 1416.33 | 1416.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 1434.10 | 1420.46 | 1418.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1415.00 | 1421.35 | 1419.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 11:15:00 | 1415.00 | 1421.35 | 1419.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 1415.00 | 1421.35 | 1419.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 1415.00 | 1421.35 | 1419.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 1409.05 | 1418.89 | 1418.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 1409.05 | 1418.89 | 1418.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 13:15:00 | 1410.20 | 1417.15 | 1417.43 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 12:15:00 | 1425.30 | 1417.84 | 1417.32 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 1413.25 | 1417.00 | 1417.26 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 1419.75 | 1417.55 | 1417.49 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 1409.45 | 1415.87 | 1416.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 10:15:00 | 1402.25 | 1409.13 | 1412.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1423.05 | 1401.04 | 1405.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 1423.05 | 1401.04 | 1405.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1423.05 | 1401.04 | 1405.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 1428.35 | 1401.04 | 1405.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1425.00 | 1405.84 | 1407.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 1424.95 | 1405.84 | 1407.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 1436.45 | 1411.96 | 1409.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1448.95 | 1430.52 | 1420.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 11:15:00 | 1563.65 | 1567.58 | 1546.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 12:00:00 | 1563.65 | 1567.58 | 1546.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1557.55 | 1561.34 | 1551.39 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 1545.00 | 1546.90 | 1547.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 1538.50 | 1545.22 | 1546.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 1551.00 | 1544.04 | 1545.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 1551.00 | 1544.04 | 1545.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1551.00 | 1544.04 | 1545.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 1551.00 | 1544.04 | 1545.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1547.20 | 1544.67 | 1545.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:30:00 | 1548.95 | 1544.67 | 1545.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1548.25 | 1545.39 | 1545.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1558.70 | 1545.39 | 1545.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1557.10 | 1547.73 | 1546.66 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 11:15:00 | 1537.05 | 1544.69 | 1545.41 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 1554.75 | 1547.31 | 1546.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 1559.60 | 1550.25 | 1548.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 1543.45 | 1548.89 | 1547.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 10:15:00 | 1543.45 | 1548.89 | 1547.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1543.45 | 1548.89 | 1547.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 1543.45 | 1548.89 | 1547.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 1544.65 | 1548.04 | 1547.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:45:00 | 1542.75 | 1548.04 | 1547.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 1548.05 | 1548.04 | 1547.45 | EMA400 retest candle locked (from upside) |

### Cycle 155 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 1543.00 | 1547.03 | 1547.04 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 1554.75 | 1547.50 | 1547.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 1561.45 | 1555.02 | 1551.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 1547.00 | 1553.62 | 1551.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 1547.00 | 1553.62 | 1551.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1547.00 | 1553.62 | 1551.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 1547.00 | 1553.62 | 1551.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1546.25 | 1552.14 | 1551.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:45:00 | 1544.70 | 1552.14 | 1551.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 1550.95 | 1551.02 | 1550.64 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 14:15:00 | 1542.15 | 1549.12 | 1549.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 1515.50 | 1541.42 | 1546.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 13:15:00 | 1470.10 | 1469.51 | 1486.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 14:00:00 | 1470.10 | 1469.51 | 1486.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1487.90 | 1473.19 | 1486.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:45:00 | 1492.15 | 1473.19 | 1486.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 1493.50 | 1477.25 | 1486.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:15:00 | 1494.35 | 1477.25 | 1486.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 1484.70 | 1480.54 | 1486.82 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1523.45 | 1491.72 | 1489.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1553.80 | 1523.24 | 1509.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1555.00 | 1562.17 | 1550.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1555.00 | 1562.17 | 1550.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1555.00 | 1562.17 | 1550.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 1559.20 | 1562.17 | 1550.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1555.50 | 1560.84 | 1550.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:00:00 | 1555.50 | 1560.84 | 1550.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1608.00 | 1617.01 | 1607.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:45:00 | 1606.40 | 1617.01 | 1607.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1605.70 | 1614.75 | 1607.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:30:00 | 1605.80 | 1614.75 | 1607.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1610.20 | 1613.84 | 1607.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:45:00 | 1613.60 | 1613.35 | 1608.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:30:00 | 1614.20 | 1613.48 | 1608.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 15:00:00 | 1614.20 | 1613.62 | 1609.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 15:15:00 | 1615.00 | 1615.63 | 1613.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 1615.00 | 1615.50 | 1613.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 1697.70 | 1615.50 | 1613.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-02 09:15:00 | 1774.96 | 1755.76 | 1733.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 1731.00 | 1750.78 | 1753.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 10:15:00 | 1709.10 | 1727.62 | 1735.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1745.20 | 1716.75 | 1724.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1745.20 | 1716.75 | 1724.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1745.20 | 1716.75 | 1724.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 1745.20 | 1716.75 | 1724.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 1747.50 | 1722.90 | 1726.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:00:00 | 1747.50 | 1722.90 | 1726.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1747.10 | 1731.40 | 1730.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1750.90 | 1738.28 | 1733.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1734.20 | 1739.53 | 1735.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 10:15:00 | 1734.20 | 1739.53 | 1735.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1734.20 | 1739.53 | 1735.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:00:00 | 1734.20 | 1739.53 | 1735.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 1749.80 | 1741.58 | 1736.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 12:15:00 | 1753.90 | 1741.58 | 1736.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:15:00 | 1753.10 | 1745.73 | 1741.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:15:00 | 1752.20 | 1746.75 | 1742.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 1755.60 | 1749.23 | 1745.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1758.90 | 1751.16 | 1747.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1772.40 | 1755.41 | 1749.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 1759.60 | 1766.43 | 1767.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1759.60 | 1766.43 | 1767.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 1752.50 | 1763.11 | 1764.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 1758.30 | 1754.84 | 1759.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 1758.30 | 1754.84 | 1759.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1758.30 | 1754.84 | 1759.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 1758.30 | 1754.84 | 1759.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 1762.00 | 1756.27 | 1759.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 1782.70 | 1756.27 | 1759.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1783.40 | 1761.70 | 1761.76 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1792.00 | 1767.76 | 1764.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 1800.00 | 1774.21 | 1767.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 1806.40 | 1807.14 | 1802.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 10:30:00 | 1804.70 | 1807.14 | 1802.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 1807.50 | 1817.64 | 1813.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:00:00 | 1807.50 | 1817.64 | 1813.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1815.10 | 1817.13 | 1813.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1815.10 | 1817.13 | 1813.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1809.00 | 1815.50 | 1813.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 1824.10 | 1815.50 | 1813.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1815.10 | 1815.42 | 1813.23 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 13:15:00 | 1809.00 | 1811.67 | 1812.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 1786.80 | 1803.87 | 1808.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 14:15:00 | 1776.60 | 1775.65 | 1785.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 15:00:00 | 1776.60 | 1775.65 | 1785.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1787.40 | 1777.12 | 1783.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 1787.40 | 1777.12 | 1783.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 1785.70 | 1778.84 | 1783.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:45:00 | 1785.80 | 1778.84 | 1783.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1770.60 | 1773.42 | 1778.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:30:00 | 1777.00 | 1773.42 | 1778.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1774.90 | 1773.71 | 1778.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:45:00 | 1779.90 | 1773.71 | 1778.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1781.00 | 1774.03 | 1777.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 1781.00 | 1774.03 | 1777.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1777.70 | 1774.76 | 1777.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 1768.00 | 1774.76 | 1777.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1792.00 | 1778.49 | 1778.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 1792.00 | 1778.49 | 1778.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 1794.70 | 1781.73 | 1780.08 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 1773.00 | 1781.11 | 1781.35 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 1785.10 | 1781.34 | 1781.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 10:15:00 | 1791.20 | 1783.31 | 1782.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 1788.80 | 1793.59 | 1789.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 11:15:00 | 1788.80 | 1793.59 | 1789.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1788.80 | 1793.59 | 1789.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1788.80 | 1793.59 | 1789.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1787.90 | 1792.45 | 1789.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1787.90 | 1792.45 | 1789.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1769.50 | 1787.86 | 1787.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1769.50 | 1787.86 | 1787.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1765.50 | 1783.39 | 1785.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 1761.30 | 1778.97 | 1783.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1761.50 | 1756.71 | 1768.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 1761.50 | 1756.71 | 1768.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1772.00 | 1759.74 | 1767.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:15:00 | 1781.00 | 1759.74 | 1767.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1793.30 | 1766.45 | 1769.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 1793.30 | 1766.45 | 1769.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1795.50 | 1772.26 | 1771.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 1799.10 | 1784.55 | 1778.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 1798.50 | 1798.71 | 1792.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 11:30:00 | 1796.00 | 1798.71 | 1792.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1791.50 | 1796.82 | 1793.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 1800.00 | 1796.82 | 1793.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1801.00 | 1797.65 | 1794.10 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 1789.20 | 1792.21 | 1792.53 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 1807.70 | 1794.96 | 1793.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 10:15:00 | 1816.30 | 1799.23 | 1795.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1801.50 | 1806.68 | 1802.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1801.50 | 1806.68 | 1802.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1801.50 | 1806.68 | 1802.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 1801.30 | 1806.68 | 1802.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1815.90 | 1808.53 | 1803.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:45:00 | 1818.60 | 1810.26 | 1804.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:15:00 | 1819.40 | 1810.26 | 1804.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 1830.40 | 1815.96 | 1809.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 1822.70 | 1848.62 | 1843.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1847.90 | 1848.48 | 1844.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:30:00 | 1862.80 | 1849.05 | 1845.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 14:15:00 | 1838.80 | 1844.56 | 1845.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 14:15:00 | 1838.80 | 1844.56 | 1845.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 15:15:00 | 1833.90 | 1842.43 | 1844.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 1843.30 | 1842.61 | 1844.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 1843.30 | 1842.61 | 1844.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1843.30 | 1842.61 | 1844.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:45:00 | 1852.90 | 1842.61 | 1844.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1834.70 | 1841.02 | 1843.18 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 13:15:00 | 1859.00 | 1846.90 | 1845.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 14:15:00 | 1863.60 | 1850.24 | 1847.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 14:15:00 | 1857.90 | 1860.46 | 1855.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 1857.90 | 1860.46 | 1855.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1857.90 | 1860.46 | 1855.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 1855.90 | 1860.46 | 1855.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1853.10 | 1858.98 | 1855.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 1841.80 | 1858.98 | 1855.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1837.00 | 1854.59 | 1853.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1837.00 | 1854.59 | 1853.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 1831.00 | 1849.87 | 1851.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 1817.60 | 1843.42 | 1848.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 1803.30 | 1799.75 | 1810.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 13:00:00 | 1803.30 | 1799.75 | 1810.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1808.90 | 1803.49 | 1809.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 1805.20 | 1803.49 | 1809.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1816.20 | 1806.55 | 1809.94 | SL hit (close>static) qty=1.00 sl=1811.80 alert=retest2 |

### Cycle 174 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 1816.20 | 1811.74 | 1811.42 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 1807.00 | 1812.16 | 1812.23 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 1818.30 | 1813.03 | 1812.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 1835.90 | 1817.29 | 1814.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 11:15:00 | 1836.10 | 1841.04 | 1834.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 12:00:00 | 1836.10 | 1841.04 | 1834.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1828.40 | 1838.51 | 1833.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 1828.40 | 1838.51 | 1833.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1838.60 | 1838.53 | 1834.32 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 1826.80 | 1831.59 | 1831.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 11:15:00 | 1823.80 | 1830.03 | 1831.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 12:15:00 | 1830.20 | 1830.07 | 1831.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 12:15:00 | 1830.20 | 1830.07 | 1831.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1830.20 | 1830.07 | 1831.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:45:00 | 1830.10 | 1830.07 | 1831.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1837.10 | 1831.47 | 1831.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:00:00 | 1837.10 | 1831.47 | 1831.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1829.60 | 1831.10 | 1831.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1817.00 | 1830.64 | 1831.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 1805.50 | 1799.91 | 1799.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 1805.50 | 1799.91 | 1799.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 12:15:00 | 1813.10 | 1806.53 | 1803.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 15:15:00 | 1805.80 | 1808.33 | 1805.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:15:00 | 1803.50 | 1808.33 | 1805.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1794.10 | 1805.49 | 1804.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 1794.10 | 1805.49 | 1804.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1792.50 | 1802.89 | 1803.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1785.70 | 1799.45 | 1801.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 15:15:00 | 1819.90 | 1798.77 | 1800.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 15:15:00 | 1819.90 | 1798.77 | 1800.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1819.90 | 1798.77 | 1800.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1836.00 | 1798.77 | 1800.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 09:15:00 | 1837.10 | 1806.43 | 1803.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 10:15:00 | 1848.50 | 1834.36 | 1822.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 1834.00 | 1842.03 | 1832.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 1834.00 | 1842.03 | 1832.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1834.00 | 1842.03 | 1832.20 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 1808.40 | 1832.02 | 1834.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1802.10 | 1826.03 | 1831.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 1813.50 | 1809.99 | 1820.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:00:00 | 1813.50 | 1809.99 | 1820.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1826.50 | 1813.87 | 1820.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1826.50 | 1813.87 | 1820.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1832.50 | 1817.60 | 1821.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1832.50 | 1817.60 | 1821.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 1836.40 | 1825.42 | 1824.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 1841.30 | 1828.60 | 1825.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 1847.00 | 1852.41 | 1845.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 1847.00 | 1852.41 | 1845.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1847.00 | 1852.41 | 1845.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:30:00 | 1860.40 | 1851.25 | 1847.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 1856.40 | 1853.24 | 1848.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 13:15:00 | 1834.80 | 1846.13 | 1846.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 1834.80 | 1846.13 | 1846.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 1829.60 | 1838.79 | 1842.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 13:15:00 | 1848.60 | 1840.16 | 1842.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 13:15:00 | 1848.60 | 1840.16 | 1842.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1848.60 | 1840.16 | 1842.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 1848.60 | 1840.16 | 1842.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1850.50 | 1842.23 | 1843.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 1850.50 | 1842.23 | 1843.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1849.50 | 1843.68 | 1843.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 1855.10 | 1845.97 | 1844.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 1840.80 | 1844.93 | 1844.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 10:15:00 | 1840.80 | 1844.93 | 1844.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1840.80 | 1844.93 | 1844.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 1840.80 | 1844.93 | 1844.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 1843.90 | 1844.73 | 1844.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:15:00 | 1848.50 | 1844.73 | 1844.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 1837.20 | 1842.83 | 1843.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 1837.20 | 1842.83 | 1843.54 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1852.00 | 1844.84 | 1844.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 1856.50 | 1847.91 | 1845.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 13:15:00 | 1848.70 | 1849.24 | 1846.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 13:15:00 | 1848.70 | 1849.24 | 1846.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1848.70 | 1849.24 | 1846.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1848.70 | 1849.24 | 1846.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1840.90 | 1847.57 | 1846.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 1840.90 | 1847.57 | 1846.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1846.10 | 1847.28 | 1846.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 1847.90 | 1847.28 | 1846.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1842.60 | 1846.34 | 1845.99 | EMA400 retest candle locked (from upside) |

### Cycle 187 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 1840.40 | 1845.15 | 1845.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 1836.10 | 1843.34 | 1844.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 13:15:00 | 1843.40 | 1842.91 | 1844.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 13:15:00 | 1843.40 | 1842.91 | 1844.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1843.40 | 1842.91 | 1844.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:45:00 | 1844.90 | 1842.91 | 1844.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1841.30 | 1842.59 | 1843.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:45:00 | 1842.60 | 1842.59 | 1843.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1905.50 | 1855.29 | 1849.47 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 1852.80 | 1857.55 | 1857.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 10:15:00 | 1849.00 | 1853.63 | 1855.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1856.90 | 1852.90 | 1854.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 14:15:00 | 1856.90 | 1852.90 | 1854.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1856.90 | 1852.90 | 1854.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 1856.90 | 1852.90 | 1854.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1862.00 | 1854.72 | 1855.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 1880.00 | 1854.72 | 1855.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1885.50 | 1860.88 | 1857.96 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 1857.00 | 1863.44 | 1864.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1846.00 | 1859.95 | 1862.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1833.40 | 1830.51 | 1840.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 1833.40 | 1830.51 | 1840.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1832.30 | 1832.19 | 1840.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:15:00 | 1829.60 | 1832.19 | 1840.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:30:00 | 1824.70 | 1828.44 | 1836.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1830.50 | 1814.03 | 1813.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 1830.50 | 1814.03 | 1813.35 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 1785.00 | 1807.69 | 1810.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 1781.70 | 1801.20 | 1807.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 1806.60 | 1799.69 | 1804.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1806.60 | 1799.69 | 1804.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1806.60 | 1799.69 | 1804.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 1779.70 | 1806.36 | 1806.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 1799.00 | 1790.21 | 1793.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 1792.10 | 1792.53 | 1794.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 1810.00 | 1798.19 | 1796.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 1810.00 | 1798.19 | 1796.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1826.50 | 1803.85 | 1799.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1821.60 | 1824.57 | 1814.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1821.60 | 1824.57 | 1814.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1821.60 | 1824.57 | 1814.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 1820.90 | 1824.57 | 1814.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 1815.50 | 1822.03 | 1815.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 1813.60 | 1822.03 | 1815.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1806.70 | 1818.96 | 1814.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 1806.70 | 1818.96 | 1814.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1813.40 | 1817.85 | 1814.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 14:30:00 | 1817.20 | 1817.04 | 1814.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 1819.00 | 1815.97 | 1814.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:45:00 | 1816.50 | 1820.97 | 1820.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 1819.10 | 1820.59 | 1820.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 1819.10 | 1820.59 | 1820.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 1814.60 | 1819.43 | 1820.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 1821.10 | 1811.39 | 1814.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 10:15:00 | 1821.10 | 1811.39 | 1814.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1821.10 | 1811.39 | 1814.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 1821.10 | 1811.39 | 1814.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1827.90 | 1814.69 | 1815.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 1827.90 | 1814.69 | 1815.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1818.50 | 1815.45 | 1815.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:45:00 | 1816.00 | 1816.06 | 1816.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 1822.80 | 1817.41 | 1816.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 1822.80 | 1817.41 | 1816.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 1840.00 | 1822.82 | 1819.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 15:15:00 | 1855.60 | 1855.83 | 1844.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:15:00 | 1847.10 | 1855.83 | 1844.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1855.50 | 1855.76 | 1845.72 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 1822.00 | 1837.84 | 1839.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1810.70 | 1829.60 | 1835.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 1813.40 | 1813.14 | 1818.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 1813.40 | 1813.14 | 1818.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1813.40 | 1813.14 | 1818.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 12:15:00 | 1803.00 | 1811.55 | 1817.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 1782.30 | 1779.19 | 1779.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 1782.30 | 1779.19 | 1779.08 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1772.10 | 1778.54 | 1778.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 1762.30 | 1775.29 | 1777.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 14:15:00 | 1773.10 | 1771.81 | 1775.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:45:00 | 1772.90 | 1771.81 | 1775.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1772.00 | 1771.85 | 1774.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1777.50 | 1771.85 | 1774.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1771.90 | 1771.86 | 1774.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1772.80 | 1771.86 | 1774.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1782.60 | 1774.01 | 1775.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 1782.60 | 1774.01 | 1775.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 1790.50 | 1777.31 | 1776.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 1799.70 | 1781.79 | 1778.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 1812.70 | 1815.81 | 1801.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 14:00:00 | 1812.70 | 1815.81 | 1801.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1821.20 | 1814.98 | 1804.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 1822.30 | 1814.98 | 1804.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1841.10 | 1814.05 | 1811.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 1825.90 | 1831.12 | 1825.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-10 09:15:00 | 2004.53 | 1992.45 | 1982.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 1981.20 | 1992.01 | 1992.09 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 2001.30 | 1992.05 | 1991.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 12:15:00 | 2005.50 | 1999.68 | 1997.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 15:15:00 | 2000.20 | 2001.25 | 1998.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 09:15:00 | 2004.30 | 2001.25 | 1998.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 2018.40 | 2020.26 | 2015.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 2075.00 | 2016.84 | 2015.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 15:00:00 | 2029.30 | 2025.84 | 2022.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:30:00 | 2029.90 | 2030.45 | 2027.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 2014.20 | 2023.94 | 2024.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 2014.20 | 2023.94 | 2024.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 2007.20 | 2018.62 | 2022.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 1971.40 | 1968.08 | 1981.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 1971.40 | 1968.08 | 1981.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1960.80 | 1967.57 | 1979.23 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 2000.60 | 1979.14 | 1977.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 11:15:00 | 2007.30 | 1984.77 | 1980.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 2018.80 | 2020.54 | 2007.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 09:15:00 | 2025.90 | 2020.54 | 2007.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 2028.00 | 2022.03 | 2009.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 11:30:00 | 2035.90 | 2024.67 | 2012.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1997.20 | 2018.92 | 2014.57 | SL hit (close<static) qty=1.00 sl=2005.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 1999.70 | 2011.19 | 2011.75 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 2023.80 | 2010.38 | 2010.32 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 2007.00 | 2010.98 | 2011.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 15:15:00 | 1997.10 | 2008.45 | 2010.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 2012.60 | 2009.28 | 2010.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 2012.60 | 2009.28 | 2010.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2012.60 | 2009.28 | 2010.32 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 2013.30 | 2011.36 | 2011.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 2025.50 | 2014.19 | 2012.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 2035.90 | 2038.95 | 2031.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 2035.90 | 2038.95 | 2031.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 2040.00 | 2039.16 | 2032.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 2043.10 | 2039.16 | 2032.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 2012.60 | 2033.85 | 2030.37 | SL hit (close<static) qty=1.00 sl=2021.00 alert=retest2 |

### Cycle 209 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 1999.60 | 2024.13 | 2026.38 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 2027.20 | 2019.60 | 2018.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2029.90 | 2024.61 | 2021.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 10:15:00 | 2020.00 | 2023.69 | 2021.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 10:15:00 | 2020.00 | 2023.69 | 2021.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 2020.00 | 2023.69 | 2021.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 2020.00 | 2023.69 | 2021.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 2016.90 | 2022.33 | 2021.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:45:00 | 2017.80 | 2022.33 | 2021.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 2021.90 | 2020.93 | 2020.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 2024.80 | 2020.93 | 2020.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 2023.40 | 2025.88 | 2023.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 2023.20 | 2025.76 | 2024.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 2006.50 | 2021.27 | 2023.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 2006.50 | 2021.27 | 2023.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 2000.00 | 2011.60 | 2015.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1996.80 | 1995.54 | 2004.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 1996.80 | 1995.54 | 2004.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 2019.70 | 2000.37 | 2006.30 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 2038.40 | 2013.02 | 2010.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 2044.00 | 2035.64 | 2027.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 2073.20 | 2076.25 | 2061.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 2073.20 | 2076.25 | 2061.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2087.90 | 2079.19 | 2066.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:00:00 | 2097.10 | 2083.81 | 2070.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 2093.30 | 2091.01 | 2078.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 12:15:00 | 2071.40 | 2079.80 | 2080.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 2071.40 | 2079.80 | 2080.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 2069.80 | 2077.80 | 2079.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 2091.70 | 2079.14 | 2079.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 2091.70 | 2079.14 | 2079.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2091.70 | 2079.14 | 2079.55 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 10:15:00 | 2085.00 | 2080.31 | 2080.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 12:15:00 | 2099.60 | 2086.20 | 2082.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 09:15:00 | 2089.90 | 2091.20 | 2086.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:15:00 | 2088.60 | 2091.20 | 2086.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 2089.90 | 2090.94 | 2087.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 2086.00 | 2090.94 | 2087.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 2083.00 | 2089.35 | 2086.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:15:00 | 2084.90 | 2089.35 | 2086.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 2080.80 | 2087.64 | 2086.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 2080.80 | 2087.64 | 2086.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 2081.10 | 2086.71 | 2086.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 2083.70 | 2086.71 | 2086.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 15:15:00 | 2076.90 | 2084.75 | 2085.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 09:15:00 | 2073.20 | 2082.44 | 2084.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 10:15:00 | 2095.60 | 2078.26 | 2079.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 10:15:00 | 2095.60 | 2078.26 | 2079.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 2095.60 | 2078.26 | 2079.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 2095.60 | 2078.26 | 2079.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 2084.90 | 2079.58 | 2080.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 2080.60 | 2079.58 | 2080.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 2079.60 | 2079.59 | 2080.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2087.00 | 2080.74 | 2080.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 2087.00 | 2080.74 | 2080.45 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 2077.10 | 2080.22 | 2080.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 2074.80 | 2079.13 | 2079.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 15:15:00 | 2079.00 | 2077.81 | 2078.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 15:15:00 | 2079.00 | 2077.81 | 2078.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 2079.00 | 2077.81 | 2078.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 2073.00 | 2077.81 | 2078.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 2062.30 | 2074.71 | 2077.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:15:00 | 2061.00 | 2074.71 | 2077.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:15:00 | 2061.60 | 2068.28 | 2073.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:15:00 | 2057.20 | 2059.89 | 2067.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 2035.00 | 2027.63 | 2027.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 2035.00 | 2027.63 | 2027.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 2064.00 | 2036.74 | 2031.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 2018.80 | 2043.13 | 2038.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 2018.80 | 2043.13 | 2038.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 2018.80 | 2043.13 | 2038.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 2007.00 | 2043.13 | 2038.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 1999.90 | 2034.48 | 2035.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 1981.10 | 2023.81 | 2030.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 11:15:00 | 2000.30 | 1995.85 | 2009.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:00:00 | 2000.30 | 1995.85 | 2009.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1994.40 | 1995.56 | 2008.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:45:00 | 2005.00 | 1995.56 | 2008.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 2008.10 | 1998.07 | 2008.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 2008.10 | 1998.07 | 2008.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1995.20 | 1997.49 | 2006.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:45:00 | 2012.90 | 1997.49 | 2006.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 2009.90 | 1999.98 | 2007.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 2003.90 | 1999.98 | 2007.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1990.10 | 1998.00 | 2005.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 1980.70 | 1990.60 | 2000.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:00:00 | 1963.00 | 1990.60 | 2000.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:45:00 | 1976.20 | 1984.45 | 1995.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:45:00 | 1980.50 | 1979.47 | 1989.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 1992.20 | 1982.02 | 1990.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 1992.20 | 1982.02 | 1990.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1996.30 | 1984.87 | 1990.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 1997.70 | 1984.87 | 1990.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1991.50 | 1986.20 | 1990.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2051.70 | 2003.57 | 1997.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2051.70 | 2003.57 | 1997.92 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 2007.90 | 2017.60 | 2018.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 1985.70 | 2011.22 | 2015.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 1997.50 | 1996.81 | 2005.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 1997.50 | 1996.81 | 2005.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2028.40 | 2002.62 | 2006.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 2028.40 | 2002.62 | 2006.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 2033.40 | 2008.77 | 2009.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 2034.20 | 2008.77 | 2009.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 2025.40 | 2012.10 | 2010.76 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 2004.90 | 2011.80 | 2012.24 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 2020.90 | 2013.67 | 2013.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 15:15:00 | 2022.10 | 2016.05 | 2014.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 2005.00 | 2019.50 | 2017.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 2005.00 | 2019.50 | 2017.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2005.00 | 2019.50 | 2017.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 2002.70 | 2019.50 | 2017.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2010.10 | 2017.62 | 2017.24 | EMA400 retest candle locked (from upside) |

### Cycle 225 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 2012.20 | 2016.53 | 2016.78 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 2020.40 | 2017.20 | 2016.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 09:15:00 | 2031.30 | 2020.47 | 2018.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 11:15:00 | 2017.00 | 2019.81 | 2018.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 2017.00 | 2019.81 | 2018.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 2017.00 | 2019.81 | 2018.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 2017.00 | 2019.81 | 2018.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 2022.50 | 2020.35 | 2018.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:00:00 | 2031.90 | 2022.66 | 2020.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 2072.10 | 2084.61 | 2085.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 227 — SELL (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 14:15:00 | 2072.10 | 2084.61 | 2085.85 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 2091.70 | 2086.29 | 2086.23 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 2077.00 | 2084.43 | 2085.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 2074.20 | 2082.38 | 2084.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 14:15:00 | 2081.60 | 2080.04 | 2082.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 14:15:00 | 2081.60 | 2080.04 | 2082.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 2081.60 | 2080.04 | 2082.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 2081.60 | 2080.04 | 2082.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 2084.90 | 2081.01 | 2083.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 2064.00 | 2081.01 | 2083.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2052.60 | 2075.33 | 2080.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 11:00:00 | 2048.20 | 2069.90 | 2077.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 2044.50 | 2063.88 | 2073.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:15:00 | 1945.79 | 1990.64 | 2019.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:15:00 | 1942.27 | 1990.64 | 2019.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 1941.50 | 1935.39 | 1964.59 | SL hit (close>ema200) qty=0.50 sl=1935.39 alert=retest2 |

### Cycle 230 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1970.90 | 1945.06 | 1941.55 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 1938.30 | 1942.34 | 1942.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 1935.00 | 1940.88 | 1942.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 1931.90 | 1930.48 | 1935.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:00:00 | 1931.90 | 1930.48 | 1935.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 1943.30 | 1933.04 | 1936.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:45:00 | 1941.70 | 1933.04 | 1936.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 1942.40 | 1934.92 | 1936.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:30:00 | 1940.40 | 1934.92 | 1936.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 1940.90 | 1936.11 | 1937.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1922.00 | 1936.11 | 1937.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 1933.60 | 1917.24 | 1916.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 232 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 1933.60 | 1917.24 | 1916.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 1935.20 | 1920.84 | 1917.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1934.00 | 1950.06 | 1939.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1934.00 | 1950.06 | 1939.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1934.00 | 1950.06 | 1939.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 1931.00 | 1950.06 | 1939.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1926.50 | 1945.35 | 1937.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 1926.50 | 1945.35 | 1937.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 233 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1907.30 | 1931.77 | 1933.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1902.40 | 1925.90 | 1930.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1844.60 | 1835.81 | 1858.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 1848.70 | 1835.81 | 1858.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1863.20 | 1842.44 | 1854.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 1865.30 | 1842.44 | 1854.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1864.80 | 1846.91 | 1855.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:45:00 | 1858.00 | 1851.69 | 1856.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 13:15:00 | 1857.40 | 1851.69 | 1856.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:00:00 | 1857.50 | 1852.85 | 1856.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:15:00 | 1765.10 | 1787.58 | 1807.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:15:00 | 1764.53 | 1787.58 | 1807.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:15:00 | 1764.62 | 1787.58 | 1807.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 14:15:00 | 1791.60 | 1786.20 | 1801.97 | SL hit (close>ema200) qty=0.50 sl=1786.20 alert=retest2 |

### Cycle 234 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 1829.60 | 1793.04 | 1788.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 1838.50 | 1802.13 | 1792.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 1900.00 | 1903.58 | 1876.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 1900.00 | 1903.58 | 1876.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1909.70 | 1922.05 | 1907.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 1895.40 | 1922.05 | 1907.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 1918.60 | 1921.36 | 1908.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:30:00 | 1907.00 | 1921.36 | 1908.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 1914.90 | 1921.13 | 1912.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 1914.90 | 1921.13 | 1912.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 1971.70 | 1971.13 | 1957.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 14:30:00 | 1973.50 | 1970.49 | 1962.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:00:00 | 1980.00 | 1972.01 | 1964.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 09:15:00 | 1900.00 | 1962.84 | 1965.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 235 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 1900.00 | 1962.84 | 1965.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1896.00 | 1917.94 | 1936.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1795.30 | 1793.28 | 1826.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:30:00 | 1784.90 | 1793.28 | 1826.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1816.40 | 1808.79 | 1819.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:00:00 | 1816.40 | 1808.79 | 1819.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1822.10 | 1811.45 | 1819.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 1822.10 | 1811.45 | 1819.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1817.70 | 1812.70 | 1819.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 1808.30 | 1813.07 | 1818.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 1826.10 | 1814.96 | 1818.18 | SL hit (close>static) qty=1.00 sl=1823.70 alert=retest2 |

### Cycle 236 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1837.10 | 1816.99 | 1815.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 1853.00 | 1835.04 | 1827.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 1871.10 | 1872.22 | 1862.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 1871.10 | 1872.22 | 1862.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-16 10:15:00 | 1191.00 | 2023-05-16 15:15:00 | 1182.55 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-05-16 11:00:00 | 1191.40 | 2023-05-16 15:15:00 | 1182.55 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2023-05-16 11:45:00 | 1192.15 | 2023-05-16 15:15:00 | 1182.55 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2023-05-22 12:30:00 | 1151.65 | 2023-05-23 09:15:00 | 1158.95 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2023-06-20 09:15:00 | 1291.70 | 2023-06-22 12:15:00 | 1290.95 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2023-07-03 13:30:00 | 1296.15 | 2023-07-04 13:15:00 | 1283.35 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2023-07-04 12:00:00 | 1296.55 | 2023-07-04 13:15:00 | 1283.35 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-07-28 09:15:00 | 1283.80 | 2023-08-07 10:15:00 | 1279.90 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2023-07-28 10:15:00 | 1291.95 | 2023-08-07 10:15:00 | 1279.90 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2023-07-28 10:45:00 | 1285.90 | 2023-08-07 10:15:00 | 1279.90 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2023-08-21 09:15:00 | 1276.55 | 2023-08-21 10:15:00 | 1288.40 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2023-08-21 14:00:00 | 1278.65 | 2023-08-22 09:15:00 | 1304.60 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2023-08-21 15:15:00 | 1277.95 | 2023-08-22 09:15:00 | 1304.60 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2023-08-23 09:15:00 | 1291.10 | 2023-08-25 10:15:00 | 1283.65 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2023-08-23 11:00:00 | 1291.30 | 2023-08-25 10:15:00 | 1283.65 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-09-04 13:15:00 | 1318.05 | 2023-09-06 13:15:00 | 1307.75 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-09-04 14:15:00 | 1318.10 | 2023-09-06 13:15:00 | 1307.75 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-09-05 14:00:00 | 1316.00 | 2023-09-06 13:15:00 | 1307.75 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-09-06 09:15:00 | 1316.95 | 2023-09-06 13:15:00 | 1307.75 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-09-13 11:45:00 | 1341.40 | 2023-09-20 13:15:00 | 1348.10 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest1 | 2023-09-22 09:15:00 | 1328.65 | 2023-09-27 14:15:00 | 1301.90 | STOP_HIT | 1.00 | 2.01% |
| SELL | retest1 | 2023-09-22 09:45:00 | 1320.10 | 2023-09-27 14:15:00 | 1301.90 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest1 | 2023-09-22 11:15:00 | 1329.25 | 2023-09-27 14:15:00 | 1301.90 | STOP_HIT | 1.00 | 2.06% |
| SELL | retest1 | 2023-09-22 12:45:00 | 1329.50 | 2023-09-27 14:15:00 | 1301.90 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2023-09-25 14:00:00 | 1289.35 | 2023-09-29 14:15:00 | 1305.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2023-09-26 14:45:00 | 1290.55 | 2023-09-29 14:15:00 | 1305.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2023-09-27 09:15:00 | 1287.90 | 2023-09-29 14:15:00 | 1305.50 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2023-09-28 14:15:00 | 1290.45 | 2023-09-29 14:15:00 | 1305.50 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2023-10-09 09:15:00 | 1278.00 | 2023-10-09 13:15:00 | 1283.75 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-10-09 11:15:00 | 1280.55 | 2023-10-09 13:15:00 | 1283.75 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2023-11-02 09:15:00 | 1355.05 | 2023-11-03 09:15:00 | 1335.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2023-11-02 13:00:00 | 1342.50 | 2023-11-03 09:15:00 | 1335.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2023-11-02 13:30:00 | 1342.35 | 2023-11-03 09:15:00 | 1335.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-11-06 10:30:00 | 1331.30 | 2023-11-07 13:15:00 | 1340.80 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-11-06 11:00:00 | 1330.00 | 2023-11-07 13:15:00 | 1340.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2023-11-06 15:15:00 | 1330.25 | 2023-11-07 13:15:00 | 1340.80 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2023-11-07 09:30:00 | 1330.65 | 2023-11-07 13:15:00 | 1340.80 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-11-07 12:15:00 | 1321.25 | 2023-11-07 13:15:00 | 1340.80 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2023-11-09 11:15:00 | 1348.50 | 2023-11-13 10:15:00 | 1335.15 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2023-11-09 14:00:00 | 1347.65 | 2023-11-13 10:15:00 | 1335.15 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-11-09 15:00:00 | 1350.85 | 2023-11-13 10:15:00 | 1335.15 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-11-21 09:15:00 | 1394.80 | 2023-11-24 09:15:00 | 1405.50 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2023-12-04 09:15:00 | 1434.40 | 2023-12-04 09:15:00 | 1422.80 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2023-12-04 10:45:00 | 1433.85 | 2023-12-15 10:15:00 | 1454.40 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2023-12-22 13:00:00 | 1398.00 | 2023-12-27 11:15:00 | 1421.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2023-12-26 10:15:00 | 1396.30 | 2023-12-27 11:15:00 | 1421.60 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-01-16 10:00:00 | 1421.35 | 2024-01-19 13:15:00 | 1434.25 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-01-17 15:00:00 | 1421.60 | 2024-01-19 13:15:00 | 1434.25 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-01-18 12:30:00 | 1421.75 | 2024-01-19 13:15:00 | 1434.25 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-01-19 10:15:00 | 1421.20 | 2024-01-19 13:15:00 | 1434.25 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-01-23 09:15:00 | 1445.05 | 2024-01-23 10:15:00 | 1414.25 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-02-06 14:00:00 | 1441.40 | 2024-02-09 13:15:00 | 1446.75 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-02-20 10:45:00 | 1498.45 | 2024-02-22 11:15:00 | 1480.45 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-02-28 12:15:00 | 1555.40 | 2024-03-04 09:15:00 | 1532.35 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-02-28 13:30:00 | 1553.55 | 2024-03-04 09:15:00 | 1532.35 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-02-29 13:45:00 | 1553.25 | 2024-03-04 09:15:00 | 1532.35 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-02-29 15:00:00 | 1553.45 | 2024-03-04 09:15:00 | 1532.35 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-03-02 09:15:00 | 1545.60 | 2024-03-04 09:15:00 | 1532.35 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-03-07 09:15:00 | 1517.60 | 2024-03-11 09:15:00 | 1525.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-03-13 09:15:00 | 1521.65 | 2024-03-13 09:15:00 | 1511.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-03-14 13:45:00 | 1493.40 | 2024-03-14 15:15:00 | 1507.85 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-03-15 09:15:00 | 1496.35 | 2024-03-22 12:15:00 | 1492.60 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2024-03-18 09:15:00 | 1497.10 | 2024-03-22 12:15:00 | 1492.60 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2024-03-28 10:00:00 | 1500.00 | 2024-04-02 09:15:00 | 1471.65 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-03-28 10:30:00 | 1500.00 | 2024-04-02 09:15:00 | 1471.65 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-03-28 12:15:00 | 1499.55 | 2024-04-02 09:15:00 | 1471.65 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-03-28 15:15:00 | 1507.30 | 2024-04-02 09:15:00 | 1471.65 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-04-12 10:30:00 | 1510.60 | 2024-04-12 11:15:00 | 1495.05 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-04-18 15:00:00 | 1453.40 | 2024-04-22 10:15:00 | 1478.55 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-04-19 09:15:00 | 1428.55 | 2024-04-22 10:15:00 | 1478.55 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-04-29 12:00:00 | 1421.50 | 2024-05-02 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-05-06 09:15:00 | 1453.95 | 2024-05-06 09:15:00 | 1437.55 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-05-13 14:45:00 | 1424.10 | 2024-05-14 15:15:00 | 1436.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-05-14 12:15:00 | 1424.45 | 2024-05-14 15:15:00 | 1436.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-05-27 11:45:00 | 1439.85 | 2024-05-27 13:15:00 | 1431.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1355.05 | 2024-06-04 09:15:00 | 1401.30 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2024-06-04 10:30:00 | 1374.90 | 2024-06-05 15:15:00 | 1385.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-06-05 12:30:00 | 1375.05 | 2024-06-05 15:15:00 | 1385.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-06-05 13:30:00 | 1373.70 | 2024-06-05 15:15:00 | 1385.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-06-12 10:30:00 | 1442.35 | 2024-06-19 14:15:00 | 1448.70 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2024-06-13 14:15:00 | 1440.35 | 2024-06-19 14:15:00 | 1448.70 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2024-06-26 09:15:00 | 1457.60 | 2024-06-26 09:15:00 | 1465.70 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-07-02 11:45:00 | 1496.55 | 2024-07-18 09:15:00 | 1646.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-02 14:00:00 | 1496.95 | 2024-07-18 09:15:00 | 1646.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-03 09:15:00 | 1498.60 | 2024-07-18 09:15:00 | 1648.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-30 11:15:00 | 1727.00 | 2024-08-05 09:15:00 | 1737.90 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2024-07-31 09:45:00 | 1733.25 | 2024-08-05 09:15:00 | 1737.90 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-08-08 13:15:00 | 1679.25 | 2024-08-08 14:15:00 | 1705.30 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-08-12 11:45:00 | 1727.80 | 2024-08-13 09:15:00 | 1681.10 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2024-08-14 15:15:00 | 1688.75 | 2024-08-20 09:15:00 | 1705.85 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-08-16 09:30:00 | 1680.65 | 2024-08-20 09:15:00 | 1705.85 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-08-16 14:30:00 | 1687.25 | 2024-08-20 09:15:00 | 1705.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-08-16 15:00:00 | 1687.65 | 2024-08-20 09:15:00 | 1705.85 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-08-19 10:15:00 | 1676.50 | 2024-08-20 09:15:00 | 1705.85 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-08-26 13:15:00 | 1799.55 | 2024-09-06 14:15:00 | 1893.45 | STOP_HIT | 1.00 | 5.22% |
| BUY | retest2 | 2024-08-27 09:45:00 | 1804.45 | 2024-09-06 14:15:00 | 1893.45 | STOP_HIT | 1.00 | 4.93% |
| BUY | retest2 | 2024-08-27 10:45:00 | 1806.10 | 2024-09-06 14:15:00 | 1893.45 | STOP_HIT | 1.00 | 4.84% |
| BUY | retest2 | 2024-09-19 14:45:00 | 1840.50 | 2024-09-25 09:15:00 | 1859.05 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2024-10-07 10:45:00 | 1787.40 | 2024-10-15 13:15:00 | 1746.40 | STOP_HIT | 1.00 | 2.29% |
| SELL | retest2 | 2024-10-07 11:45:00 | 1788.35 | 2024-10-15 13:15:00 | 1746.40 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2024-10-21 10:30:00 | 1710.00 | 2024-10-23 12:15:00 | 1718.95 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-10-21 12:00:00 | 1710.00 | 2024-10-23 12:15:00 | 1718.95 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-10-21 14:00:00 | 1709.50 | 2024-10-23 12:15:00 | 1718.95 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-10-21 15:15:00 | 1707.00 | 2024-10-23 12:15:00 | 1718.95 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-11-04 10:30:00 | 1601.45 | 2024-11-05 13:15:00 | 1639.30 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-11-04 11:30:00 | 1601.85 | 2024-11-05 13:15:00 | 1639.30 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-11-04 12:15:00 | 1600.55 | 2024-11-05 14:15:00 | 1631.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-11-04 12:45:00 | 1600.25 | 2024-11-05 14:15:00 | 1631.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-11-05 11:30:00 | 1602.00 | 2024-11-05 14:15:00 | 1631.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-11-05 12:15:00 | 1601.40 | 2024-11-05 14:15:00 | 1631.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-11-08 11:15:00 | 1584.15 | 2024-11-19 12:15:00 | 1504.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 11:15:00 | 1584.15 | 2024-11-22 11:15:00 | 1490.20 | STOP_HIT | 0.50 | 5.93% |
| SELL | retest2 | 2024-12-03 10:30:00 | 1419.10 | 2024-12-03 13:15:00 | 1444.55 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-12-11 10:30:00 | 1467.50 | 2024-12-12 09:15:00 | 1438.20 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-12-11 11:30:00 | 1464.20 | 2024-12-12 09:15:00 | 1438.20 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-12-16 10:15:00 | 1420.75 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2024-12-16 12:00:00 | 1421.65 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2024-12-16 15:00:00 | 1421.95 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2024-12-17 10:45:00 | 1420.75 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2024-12-19 15:15:00 | 1402.30 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-12-20 09:30:00 | 1398.75 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-12-20 12:45:00 | 1403.40 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-12-20 15:00:00 | 1394.30 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-12-23 13:00:00 | 1396.55 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-12-31 09:15:00 | 1387.20 | 2025-01-02 09:15:00 | 1402.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-12-31 12:30:00 | 1390.00 | 2025-01-02 09:15:00 | 1402.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-31 13:30:00 | 1390.10 | 2025-01-02 09:15:00 | 1402.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-31 14:15:00 | 1390.00 | 2025-01-02 09:15:00 | 1402.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-01-07 09:15:00 | 1446.65 | 2025-01-20 13:15:00 | 1497.70 | STOP_HIT | 1.00 | 3.53% |
| SELL | retest1 | 2025-01-23 09:15:00 | 1450.50 | 2025-01-29 09:15:00 | 1438.25 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest1 | 2025-01-23 11:15:00 | 1457.00 | 2025-01-29 09:15:00 | 1438.25 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest1 | 2025-01-23 12:15:00 | 1457.10 | 2025-01-29 09:15:00 | 1438.25 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest1 | 2025-01-23 14:15:00 | 1456.85 | 2025-01-29 09:15:00 | 1438.25 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2025-01-28 15:00:00 | 1420.20 | 2025-01-29 09:15:00 | 1438.25 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-02-17 10:30:00 | 1479.60 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-02-17 11:30:00 | 1479.80 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-02-17 13:15:00 | 1478.85 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-02-18 13:30:00 | 1479.55 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-02-20 09:15:00 | 1483.00 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-02-20 10:15:00 | 1482.55 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-04-23 12:45:00 | 1613.60 | 2025-05-02 09:15:00 | 1774.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-23 13:30:00 | 1614.20 | 2025-05-02 09:15:00 | 1775.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-23 15:00:00 | 1614.20 | 2025-05-02 09:15:00 | 1775.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-24 15:15:00 | 1615.00 | 2025-05-02 09:15:00 | 1776.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-25 09:15:00 | 1697.70 | 2025-05-06 13:15:00 | 1731.00 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest2 | 2025-05-13 12:15:00 | 1753.90 | 2025-05-20 14:15:00 | 1759.60 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-05-14 11:15:00 | 1753.10 | 2025-05-20 14:15:00 | 1759.60 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-05-14 12:15:00 | 1752.20 | 2025-05-20 14:15:00 | 1759.60 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-05-15 10:30:00 | 1755.60 | 2025-05-20 14:15:00 | 1759.60 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-05-15 13:00:00 | 1772.40 | 2025-05-20 14:15:00 | 1759.60 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-23 11:45:00 | 1818.60 | 2025-06-30 14:15:00 | 1838.80 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2025-06-23 12:15:00 | 1819.40 | 2025-06-30 14:15:00 | 1838.80 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1830.40 | 2025-06-30 14:15:00 | 1838.80 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-06-27 10:00:00 | 1822.70 | 2025-06-30 14:15:00 | 1838.80 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2025-06-27 14:30:00 | 1862.80 | 2025-06-30 14:15:00 | 1838.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-07-08 09:15:00 | 1805.20 | 2025-07-08 10:15:00 | 1816.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-17 09:15:00 | 1817.00 | 2025-07-22 12:15:00 | 1805.50 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2025-08-07 14:30:00 | 1860.40 | 2025-08-08 13:15:00 | 1834.80 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-08-08 10:15:00 | 1856.40 | 2025-08-08 13:15:00 | 1834.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-12 12:15:00 | 1848.50 | 2025-08-12 14:15:00 | 1837.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-28 12:15:00 | 1829.60 | 2025-09-04 09:15:00 | 1830.50 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-08-28 14:30:00 | 1824.70 | 2025-09-04 09:15:00 | 1830.50 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-09-08 09:15:00 | 1779.70 | 2025-09-09 15:15:00 | 1810.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-09-09 11:45:00 | 1799.00 | 2025-09-09 15:15:00 | 1810.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-09 13:15:00 | 1792.10 | 2025-09-09 15:15:00 | 1810.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-11 14:30:00 | 1817.20 | 2025-09-16 15:15:00 | 1819.10 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-09-12 10:45:00 | 1819.00 | 2025-09-16 15:15:00 | 1819.10 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-09-16 14:45:00 | 1816.50 | 2025-09-16 15:15:00 | 1819.10 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-09-18 13:45:00 | 1816.00 | 2025-09-18 14:15:00 | 1822.80 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-09-26 12:15:00 | 1803.00 | 2025-10-07 14:15:00 | 1782.30 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2025-10-13 10:15:00 | 1822.30 | 2025-11-10 09:15:00 | 2004.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1841.10 | 2025-11-10 09:15:00 | 2008.49 | TARGET_HIT | 1.00 | 9.09% |
| BUY | retest2 | 2025-10-16 11:00:00 | 1825.90 | 2025-11-13 10:15:00 | 1981.20 | STOP_HIT | 1.00 | 8.51% |
| BUY | retest2 | 2025-11-24 15:15:00 | 2075.00 | 2025-11-27 10:15:00 | 2014.20 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-11-25 15:00:00 | 2029.30 | 2025-11-27 10:15:00 | 2014.20 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-11-26 14:30:00 | 2029.90 | 2025-11-27 10:15:00 | 2014.20 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-08 11:30:00 | 2035.90 | 2025-12-09 09:15:00 | 1997.20 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-12-17 09:15:00 | 2043.10 | 2025-12-17 09:15:00 | 2012.60 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-12-22 15:15:00 | 2024.80 | 2025-12-26 10:15:00 | 2006.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-12-23 14:45:00 | 2023.40 | 2025-12-26 10:15:00 | 2006.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-12-26 09:15:00 | 2023.20 | 2025-12-26 10:15:00 | 2006.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-06 12:00:00 | 2097.10 | 2026-01-09 12:15:00 | 2071.40 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-01-07 09:15:00 | 2093.30 | 2026-01-09 12:15:00 | 2071.40 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-01-16 12:15:00 | 2080.60 | 2026-01-19 09:15:00 | 2087.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-01-16 13:00:00 | 2079.60 | 2026-01-19 09:15:00 | 2087.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-01-20 10:15:00 | 2061.00 | 2026-01-27 15:15:00 | 2035.00 | STOP_HIT | 1.00 | 1.26% |
| SELL | retest2 | 2026-01-20 13:15:00 | 2061.60 | 2026-01-27 15:15:00 | 2035.00 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2026-01-21 10:15:00 | 2057.20 | 2026-01-27 15:15:00 | 2035.00 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2026-02-01 11:30:00 | 1980.70 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2026-02-01 12:00:00 | 1963.00 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2026-02-01 14:45:00 | 1976.20 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2026-02-02 10:45:00 | 1980.50 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2026-02-13 14:00:00 | 2031.90 | 2026-02-25 14:15:00 | 2072.10 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2026-02-27 11:00:00 | 2048.20 | 2026-03-04 11:15:00 | 1945.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 11:45:00 | 2044.50 | 2026-03-04 11:15:00 | 1942.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 11:00:00 | 2048.20 | 2026-03-05 14:15:00 | 1941.50 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2026-02-27 11:45:00 | 2044.50 | 2026-03-05 14:15:00 | 1941.50 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1922.00 | 2026-03-17 13:15:00 | 1933.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-03-25 12:45:00 | 1858.00 | 2026-04-01 11:15:00 | 1765.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 13:15:00 | 1857.40 | 2026-04-01 11:15:00 | 1764.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:00:00 | 1857.50 | 2026-04-01 11:15:00 | 1764.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 12:45:00 | 1858.00 | 2026-04-01 14:15:00 | 1791.60 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2026-03-25 13:15:00 | 1857.40 | 2026-04-01 14:15:00 | 1791.60 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2026-03-25 14:00:00 | 1857.50 | 2026-04-01 14:15:00 | 1791.60 | STOP_HIT | 0.50 | 3.55% |
| BUY | retest2 | 2026-04-17 14:30:00 | 1973.50 | 2026-04-21 09:15:00 | 1900.00 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2026-04-20 10:00:00 | 1980.00 | 2026-04-21 09:15:00 | 1900.00 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2026-04-28 15:00:00 | 1808.30 | 2026-04-29 09:15:00 | 1826.10 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-04-29 12:15:00 | 1811.80 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-04-29 14:15:00 | 1811.20 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1810.70 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1795.00 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -2.35% |
