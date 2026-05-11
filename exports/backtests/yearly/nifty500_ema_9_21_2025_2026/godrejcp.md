# Godrej Consumer Products Ltd. (GODREJCP)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1041.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 74 |
| ALERT1 | 50 |
| ALERT2 | 50 |
| ALERT2_SKIP | 24 |
| ALERT3 | 125 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 52 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 40
- **Target hits / Stop hits / Partials:** 2 / 53 / 4
- **Avg / median % per leg:** 0.50% / -0.79%
- **Sum % (uncompounded):** 29.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 8 | 47.1% | 2 | 15 | 0 | 1.22% | 20.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.85% | -1.8% |
| BUY @ 3rd Alert (retest2) | 16 | 8 | 50.0% | 2 | 14 | 0 | 1.41% | 22.5% |
| SELL (all) | 42 | 11 | 26.2% | 0 | 38 | 4 | 0.21% | 8.9% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.37% | -0.7% |
| SELL @ 3rd Alert (retest2) | 40 | 10 | 25.0% | 0 | 36 | 4 | 0.24% | 9.6% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.86% | -2.6% |
| retest2 (combined) | 56 | 18 | 32.1% | 2 | 50 | 4 | 0.57% | 32.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1270.40 | 1250.31 | 1248.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1280.00 | 1259.25 | 1252.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 1274.00 | 1276.58 | 1266.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 11:45:00 | 1270.40 | 1276.58 | 1266.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1265.50 | 1272.61 | 1266.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:15:00 | 1267.90 | 1272.61 | 1266.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1267.90 | 1271.67 | 1266.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:15:00 | 1268.00 | 1271.67 | 1266.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1263.70 | 1270.07 | 1266.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:45:00 | 1262.00 | 1270.07 | 1266.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 1266.50 | 1269.36 | 1266.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:30:00 | 1268.40 | 1269.36 | 1266.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 1276.20 | 1270.73 | 1267.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:15:00 | 1278.10 | 1270.73 | 1267.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:00:00 | 1284.20 | 1276.86 | 1271.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:30:00 | 1276.80 | 1276.53 | 1272.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:45:00 | 1278.50 | 1276.27 | 1272.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1291.00 | 1295.66 | 1291.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 1291.00 | 1295.66 | 1291.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 1291.10 | 1294.75 | 1291.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 1289.40 | 1294.75 | 1291.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1286.90 | 1293.18 | 1291.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 1286.90 | 1293.18 | 1291.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1290.50 | 1292.64 | 1291.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:45:00 | 1287.80 | 1292.64 | 1291.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1281.60 | 1290.43 | 1290.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 1282.30 | 1290.43 | 1290.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 1271.90 | 1286.73 | 1288.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1271.90 | 1286.73 | 1288.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1262.30 | 1281.84 | 1286.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1278.30 | 1275.34 | 1281.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1278.30 | 1275.34 | 1281.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1278.30 | 1275.34 | 1281.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1278.30 | 1275.34 | 1281.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1286.20 | 1277.53 | 1281.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 13:00:00 | 1286.20 | 1277.53 | 1281.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 1283.90 | 1278.81 | 1281.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 14:30:00 | 1278.80 | 1279.04 | 1281.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 15:15:00 | 1278.80 | 1275.90 | 1275.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 1278.80 | 1275.90 | 1275.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1283.00 | 1277.32 | 1276.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 1278.70 | 1279.76 | 1278.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 13:15:00 | 1278.70 | 1279.76 | 1278.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1278.70 | 1279.76 | 1278.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 1278.70 | 1279.76 | 1278.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 1285.40 | 1280.89 | 1278.69 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 1269.30 | 1276.03 | 1276.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 1264.70 | 1273.76 | 1275.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 1268.50 | 1242.15 | 1246.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 1268.50 | 1242.15 | 1246.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1268.50 | 1242.15 | 1246.66 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 1267.50 | 1251.47 | 1250.38 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 1239.90 | 1252.50 | 1252.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 14:15:00 | 1238.00 | 1246.13 | 1249.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 10:15:00 | 1230.70 | 1229.27 | 1235.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-05 11:00:00 | 1230.70 | 1229.27 | 1235.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1223.70 | 1214.74 | 1220.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:45:00 | 1223.20 | 1214.74 | 1220.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 1214.20 | 1214.63 | 1219.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 14:15:00 | 1211.80 | 1217.23 | 1218.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 1226.50 | 1218.69 | 1218.78 | SL hit (close>static) qty=1.00 sl=1224.20 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 1222.20 | 1219.39 | 1219.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 11:15:00 | 1229.00 | 1221.31 | 1219.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 1216.30 | 1222.05 | 1221.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1216.30 | 1222.05 | 1221.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1216.30 | 1222.05 | 1221.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1215.40 | 1222.05 | 1221.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 1205.80 | 1218.80 | 1219.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1200.70 | 1212.70 | 1216.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 1189.30 | 1189.00 | 1197.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 1189.30 | 1189.00 | 1197.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1190.50 | 1188.25 | 1193.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 1196.90 | 1188.25 | 1193.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1188.50 | 1188.30 | 1193.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 1183.90 | 1188.30 | 1193.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 14:15:00 | 1197.80 | 1190.14 | 1192.33 | SL hit (close>static) qty=1.00 sl=1193.50 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 13:15:00 | 1201.00 | 1192.77 | 1192.72 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 1190.00 | 1193.61 | 1193.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 1186.50 | 1190.92 | 1192.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 1183.90 | 1178.23 | 1182.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1183.90 | 1178.23 | 1182.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1183.90 | 1178.23 | 1182.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 1182.50 | 1178.23 | 1182.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1186.70 | 1179.92 | 1183.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:30:00 | 1187.00 | 1179.92 | 1183.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 1181.00 | 1180.36 | 1182.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:45:00 | 1182.00 | 1180.36 | 1182.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1175.00 | 1179.29 | 1182.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 15:00:00 | 1173.60 | 1178.15 | 1181.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:00:00 | 1171.80 | 1176.68 | 1180.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 1185.10 | 1179.79 | 1180.63 | SL hit (close>static) qty=1.00 sl=1182.30 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 15:15:00 | 1185.50 | 1181.69 | 1181.40 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 1176.20 | 1181.06 | 1181.22 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 14:15:00 | 1184.60 | 1181.63 | 1181.42 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 1178.00 | 1181.37 | 1181.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 12:15:00 | 1175.70 | 1180.24 | 1180.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 15:15:00 | 1181.00 | 1178.87 | 1179.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 15:15:00 | 1181.00 | 1178.87 | 1179.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1181.00 | 1178.87 | 1179.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1174.50 | 1178.87 | 1179.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1176.80 | 1178.45 | 1179.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:30:00 | 1170.30 | 1176.04 | 1177.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 11:45:00 | 1168.40 | 1174.63 | 1176.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:45:00 | 1169.40 | 1172.42 | 1175.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 11:15:00 | 1178.10 | 1175.21 | 1175.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 1178.10 | 1175.21 | 1175.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 13:15:00 | 1179.10 | 1176.50 | 1175.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 1171.20 | 1175.44 | 1175.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 14:15:00 | 1171.20 | 1175.44 | 1175.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1171.20 | 1175.44 | 1175.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 1171.20 | 1175.44 | 1175.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 1173.80 | 1175.11 | 1175.15 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 12:15:00 | 1178.40 | 1175.19 | 1175.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 13:15:00 | 1183.50 | 1176.85 | 1175.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1285.60 | 1289.25 | 1272.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:15:00 | 1297.60 | 1287.17 | 1278.43 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1285.50 | 1289.40 | 1283.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:30:00 | 1287.90 | 1288.48 | 1283.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 1288.20 | 1288.48 | 1283.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 1273.60 | 1285.46 | 1283.04 | SL hit (close<ema400) qty=1.00 sl=1283.04 alert=retest1 |

### Cycle 18 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 1271.20 | 1281.50 | 1281.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 13:15:00 | 1265.70 | 1270.52 | 1274.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 15:15:00 | 1271.60 | 1270.72 | 1273.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 15:15:00 | 1271.60 | 1270.72 | 1273.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1271.60 | 1270.72 | 1273.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1265.80 | 1270.72 | 1273.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 1278.00 | 1266.51 | 1268.68 | SL hit (close>static) qty=1.00 sl=1274.20 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 11:15:00 | 1281.20 | 1271.60 | 1270.76 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 1260.40 | 1270.83 | 1271.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 11:15:00 | 1254.40 | 1267.54 | 1269.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 14:15:00 | 1241.80 | 1241.39 | 1247.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 14:30:00 | 1239.80 | 1241.39 | 1247.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1246.80 | 1242.47 | 1247.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 1236.50 | 1242.47 | 1247.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 1234.90 | 1218.46 | 1217.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 09:15:00 | 1234.90 | 1218.46 | 1217.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 10:15:00 | 1250.60 | 1224.89 | 1220.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 15:15:00 | 1257.60 | 1263.34 | 1251.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:15:00 | 1240.30 | 1263.34 | 1251.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1242.00 | 1259.07 | 1250.45 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1232.10 | 1246.80 | 1247.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 11:15:00 | 1226.40 | 1240.27 | 1243.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 1210.70 | 1205.58 | 1214.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 13:15:00 | 1210.70 | 1205.58 | 1214.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1210.70 | 1205.58 | 1214.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 1210.70 | 1205.58 | 1214.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1220.10 | 1208.49 | 1214.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:30:00 | 1223.00 | 1208.49 | 1214.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1222.00 | 1211.19 | 1215.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1212.30 | 1211.19 | 1215.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1199.80 | 1193.62 | 1199.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 1199.80 | 1193.62 | 1199.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1200.80 | 1195.06 | 1199.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:30:00 | 1198.80 | 1195.06 | 1199.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1200.00 | 1196.04 | 1199.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 1205.00 | 1196.04 | 1199.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1204.90 | 1197.82 | 1200.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 1204.90 | 1197.82 | 1200.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1190.60 | 1196.37 | 1199.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 1187.40 | 1195.52 | 1197.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 1189.10 | 1196.27 | 1197.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 1186.00 | 1196.27 | 1197.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1205.50 | 1190.07 | 1192.63 | SL hit (close>static) qty=1.00 sl=1205.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 1218.30 | 1195.72 | 1194.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 1221.60 | 1200.90 | 1197.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 1209.10 | 1209.10 | 1202.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 1209.10 | 1209.10 | 1202.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 1212.30 | 1210.10 | 1205.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:30:00 | 1215.60 | 1211.80 | 1207.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 1236.70 | 1254.42 | 1256.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 1236.70 | 1254.42 | 1256.18 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 1261.20 | 1253.35 | 1252.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1274.40 | 1257.97 | 1255.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1272.00 | 1274.94 | 1267.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1272.00 | 1274.94 | 1267.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1279.40 | 1275.83 | 1268.67 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 1247.20 | 1267.94 | 1268.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 1241.80 | 1252.34 | 1259.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1241.20 | 1236.86 | 1245.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:45:00 | 1242.00 | 1236.86 | 1245.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1233.10 | 1233.04 | 1239.18 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 1249.00 | 1241.04 | 1240.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 1251.70 | 1243.41 | 1242.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 1244.00 | 1255.04 | 1251.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 1244.00 | 1255.04 | 1251.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1244.00 | 1255.04 | 1251.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 1238.50 | 1255.04 | 1251.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1242.10 | 1252.45 | 1250.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 1245.60 | 1251.06 | 1249.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:45:00 | 1245.20 | 1250.11 | 1249.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 14:15:00 | 1248.30 | 1249.11 | 1249.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 14:15:00 | 1248.30 | 1249.11 | 1249.12 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 1253.00 | 1249.88 | 1249.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 11:15:00 | 1256.20 | 1251.41 | 1250.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 1253.70 | 1254.42 | 1252.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1253.70 | 1254.42 | 1252.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1253.70 | 1254.42 | 1252.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 1252.50 | 1254.42 | 1252.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1253.70 | 1254.27 | 1252.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 1254.30 | 1254.27 | 1252.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1246.20 | 1252.66 | 1251.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 1246.20 | 1252.66 | 1251.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 1236.20 | 1249.37 | 1250.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 14:15:00 | 1230.00 | 1243.18 | 1247.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 11:15:00 | 1235.80 | 1235.80 | 1241.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 11:30:00 | 1235.80 | 1235.80 | 1241.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1241.30 | 1237.31 | 1241.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1241.30 | 1237.31 | 1241.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1242.60 | 1238.37 | 1241.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 1233.70 | 1238.37 | 1241.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1242.90 | 1238.02 | 1237.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 1242.90 | 1238.02 | 1237.56 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 1232.20 | 1236.70 | 1237.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 1216.00 | 1231.51 | 1234.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 1196.30 | 1196.18 | 1208.91 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 14:15:00 | 1191.00 | 1195.84 | 1207.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1176.00 | 1171.64 | 1177.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 1183.50 | 1171.64 | 1177.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1178.00 | 1172.91 | 1177.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-29 15:15:00 | 1178.00 | 1172.91 | 1177.63 | SL hit (close>ema400) qty=1.00 sl=1177.63 alert=retest1 |

### Cycle 33 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 1116.80 | 1112.49 | 1112.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 15:15:00 | 1121.00 | 1115.25 | 1113.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 1130.50 | 1131.66 | 1125.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:45:00 | 1129.00 | 1131.66 | 1125.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1129.00 | 1134.73 | 1130.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1129.00 | 1134.73 | 1130.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1133.40 | 1134.46 | 1131.10 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1119.40 | 1127.98 | 1129.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 1115.70 | 1123.07 | 1125.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 12:15:00 | 1107.40 | 1107.18 | 1112.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 13:00:00 | 1107.40 | 1107.18 | 1112.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 1112.90 | 1108.97 | 1112.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:30:00 | 1115.80 | 1108.97 | 1112.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 1114.00 | 1109.97 | 1112.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 1114.10 | 1109.97 | 1112.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1119.20 | 1113.04 | 1113.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1119.20 | 1113.04 | 1113.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 1120.00 | 1114.43 | 1114.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1171.60 | 1128.43 | 1121.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1158.40 | 1161.49 | 1145.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 10:00:00 | 1158.40 | 1161.49 | 1145.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1139.90 | 1156.13 | 1150.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1139.90 | 1156.13 | 1150.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1134.70 | 1151.84 | 1148.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 1134.70 | 1151.84 | 1148.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 1144.80 | 1148.56 | 1147.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:45:00 | 1143.90 | 1148.56 | 1147.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1144.30 | 1147.71 | 1147.53 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 1143.50 | 1146.87 | 1147.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1132.60 | 1144.01 | 1145.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 1131.30 | 1128.64 | 1135.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 10:15:00 | 1131.30 | 1128.64 | 1135.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1131.30 | 1128.64 | 1135.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:45:00 | 1135.90 | 1128.64 | 1135.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1135.40 | 1129.99 | 1135.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 1138.00 | 1129.99 | 1135.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 1132.00 | 1130.39 | 1134.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:15:00 | 1130.20 | 1130.39 | 1134.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:15:00 | 1130.20 | 1129.83 | 1132.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:15:00 | 1130.70 | 1130.62 | 1133.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 1139.10 | 1131.50 | 1132.52 | SL hit (close>static) qty=1.00 sl=1138.30 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 1144.80 | 1134.16 | 1133.64 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 1129.80 | 1133.29 | 1133.29 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1140.00 | 1134.63 | 1133.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 1141.00 | 1135.90 | 1134.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 12:15:00 | 1134.20 | 1135.56 | 1134.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 12:15:00 | 1134.20 | 1135.56 | 1134.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1134.20 | 1135.56 | 1134.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 1134.20 | 1135.56 | 1134.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 1135.30 | 1135.51 | 1134.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 15:15:00 | 1139.00 | 1135.61 | 1134.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 1127.90 | 1134.61 | 1134.45 | SL hit (close<static) qty=1.00 sl=1130.70 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 1132.80 | 1134.25 | 1134.30 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 1136.20 | 1134.64 | 1134.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 13:15:00 | 1139.30 | 1135.96 | 1135.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 1132.10 | 1135.19 | 1134.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 1132.10 | 1135.19 | 1134.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1132.10 | 1135.19 | 1134.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1132.10 | 1135.19 | 1134.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1135.00 | 1135.15 | 1134.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1132.40 | 1135.15 | 1134.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1130.20 | 1134.16 | 1134.44 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 1138.40 | 1135.37 | 1134.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 13:15:00 | 1141.20 | 1136.53 | 1135.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 15:15:00 | 1149.70 | 1151.10 | 1145.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 09:15:00 | 1147.70 | 1151.10 | 1145.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1143.20 | 1149.52 | 1145.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 1141.80 | 1149.52 | 1145.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1139.60 | 1147.53 | 1145.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1141.60 | 1147.53 | 1145.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 1145.30 | 1146.52 | 1144.99 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 1140.00 | 1144.19 | 1144.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1136.30 | 1142.61 | 1143.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1147.30 | 1142.43 | 1143.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 1147.30 | 1142.43 | 1143.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1147.30 | 1142.43 | 1143.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 1147.30 | 1142.43 | 1143.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1145.00 | 1142.94 | 1143.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:15:00 | 1142.00 | 1142.94 | 1143.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 1142.80 | 1127.93 | 1127.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 1142.80 | 1127.93 | 1127.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1148.60 | 1136.21 | 1131.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 1147.20 | 1148.19 | 1142.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:45:00 | 1146.20 | 1148.19 | 1142.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1141.10 | 1146.10 | 1142.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 1140.80 | 1146.10 | 1142.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1143.70 | 1145.62 | 1142.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:45:00 | 1142.90 | 1145.62 | 1142.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1150.80 | 1147.16 | 1143.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 1148.70 | 1147.16 | 1143.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 1142.90 | 1146.45 | 1143.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:45:00 | 1142.30 | 1146.45 | 1143.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 1144.60 | 1146.08 | 1143.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 1145.10 | 1146.08 | 1143.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 1144.60 | 1145.78 | 1144.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:45:00 | 1144.00 | 1145.78 | 1144.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1145.60 | 1145.74 | 1144.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:45:00 | 1145.20 | 1145.74 | 1144.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1145.00 | 1145.60 | 1144.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 1136.00 | 1145.60 | 1144.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 1133.10 | 1143.10 | 1143.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 1130.00 | 1140.48 | 1142.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 1136.50 | 1133.23 | 1136.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 1136.50 | 1133.23 | 1136.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1136.50 | 1133.23 | 1136.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 1138.40 | 1133.23 | 1136.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1137.00 | 1133.98 | 1136.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 1137.30 | 1133.98 | 1136.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1132.10 | 1133.60 | 1136.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 1120.40 | 1133.74 | 1135.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:45:00 | 1125.20 | 1128.33 | 1129.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 1133.60 | 1130.10 | 1129.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 1133.60 | 1130.10 | 1129.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 1136.90 | 1132.05 | 1130.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1125.10 | 1130.66 | 1130.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1125.10 | 1130.66 | 1130.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1125.10 | 1130.66 | 1130.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1125.10 | 1130.66 | 1130.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1125.30 | 1129.59 | 1129.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1122.70 | 1128.21 | 1129.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 15:15:00 | 1123.40 | 1122.74 | 1125.79 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:15:00 | 1111.20 | 1122.74 | 1125.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1131.60 | 1123.91 | 1125.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 1131.60 | 1123.91 | 1125.76 | SL hit (close>ema400) qty=1.00 sl=1125.76 alert=retest1 |

### Cycle 49 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 1130.20 | 1127.31 | 1127.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 1136.30 | 1129.75 | 1128.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1129.90 | 1132.04 | 1130.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 1129.90 | 1132.04 | 1130.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1129.90 | 1132.04 | 1130.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 1129.90 | 1132.04 | 1130.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1126.10 | 1130.86 | 1129.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1135.10 | 1130.86 | 1129.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:45:00 | 1131.80 | 1132.28 | 1130.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-01 14:15:00 | 1244.98 | 1232.92 | 1223.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 1236.30 | 1241.59 | 1241.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 1228.10 | 1238.33 | 1239.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1240.00 | 1234.07 | 1236.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 11:15:00 | 1240.00 | 1234.07 | 1236.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 1240.00 | 1234.07 | 1236.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:00:00 | 1240.00 | 1234.07 | 1236.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1240.20 | 1235.30 | 1236.66 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1249.80 | 1238.20 | 1237.85 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 12:15:00 | 1235.30 | 1237.42 | 1237.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 13:15:00 | 1231.90 | 1236.32 | 1237.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 1238.90 | 1236.84 | 1237.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 1238.90 | 1236.84 | 1237.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1238.90 | 1236.84 | 1237.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 1238.90 | 1236.84 | 1237.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1233.00 | 1236.07 | 1236.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1231.90 | 1234.21 | 1235.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 1231.30 | 1234.38 | 1235.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:00:00 | 1230.50 | 1233.61 | 1235.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:00:00 | 1230.70 | 1233.02 | 1234.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1245.60 | 1233.17 | 1234.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1245.60 | 1233.17 | 1234.22 | SL hit (close>static) qty=1.00 sl=1241.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 1236.10 | 1234.96 | 1234.93 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1228.40 | 1235.39 | 1235.40 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 09:15:00 | 1251.90 | 1236.46 | 1235.50 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1232.90 | 1236.54 | 1236.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 12:15:00 | 1225.20 | 1232.97 | 1235.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 1227.90 | 1227.68 | 1231.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 1233.00 | 1227.68 | 1231.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1232.80 | 1228.70 | 1231.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1235.80 | 1228.70 | 1231.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1226.50 | 1228.26 | 1231.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1224.00 | 1228.26 | 1231.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 13:15:00 | 1240.00 | 1230.10 | 1231.34 | SL hit (close>static) qty=1.00 sl=1234.20 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 1244.90 | 1233.06 | 1232.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 1247.80 | 1236.01 | 1233.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 1239.70 | 1239.76 | 1236.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 14:00:00 | 1239.70 | 1239.76 | 1236.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1242.70 | 1240.45 | 1237.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 1197.40 | 1240.45 | 1237.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 1178.30 | 1228.02 | 1232.27 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 1165.00 | 1158.35 | 1158.12 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 09:15:00 | 1155.80 | 1157.84 | 1157.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-03 12:15:00 | 1147.20 | 1155.53 | 1156.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 13:15:00 | 1148.20 | 1147.71 | 1151.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 14:00:00 | 1148.20 | 1147.71 | 1151.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1155.70 | 1149.31 | 1151.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:45:00 | 1156.60 | 1149.31 | 1151.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 1153.60 | 1150.17 | 1151.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 1170.80 | 1150.17 | 1151.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 09:15:00 | 1164.60 | 1153.05 | 1152.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 11:15:00 | 1175.10 | 1166.19 | 1161.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 1202.40 | 1203.89 | 1196.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 1202.40 | 1203.89 | 1196.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1202.40 | 1203.89 | 1196.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 1200.00 | 1203.89 | 1196.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1206.10 | 1204.33 | 1197.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:30:00 | 1209.90 | 1204.10 | 1198.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1192.90 | 1200.87 | 1198.80 | SL hit (close<static) qty=1.00 sl=1196.60 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1192.00 | 1197.07 | 1197.73 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1216.70 | 1201.55 | 1199.68 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1200.50 | 1206.71 | 1207.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1187.50 | 1201.21 | 1204.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 1198.60 | 1197.39 | 1201.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 10:45:00 | 1198.40 | 1197.39 | 1201.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 1206.40 | 1199.19 | 1202.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 1206.30 | 1199.19 | 1202.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1204.30 | 1200.21 | 1202.41 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 1213.90 | 1205.17 | 1204.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 11:15:00 | 1219.10 | 1209.25 | 1206.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 1233.90 | 1234.01 | 1228.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 1233.90 | 1234.01 | 1228.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1233.90 | 1234.01 | 1228.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 1233.70 | 1234.01 | 1228.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1214.10 | 1230.84 | 1229.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 1214.10 | 1230.84 | 1229.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 1219.20 | 1228.51 | 1228.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1184.00 | 1214.84 | 1221.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1133.60 | 1133.56 | 1151.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 1138.00 | 1133.56 | 1151.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1117.00 | 1094.04 | 1108.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 1117.00 | 1094.04 | 1108.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 1118.70 | 1098.98 | 1109.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 1118.70 | 1098.98 | 1109.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 1115.70 | 1102.32 | 1109.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 1112.60 | 1102.32 | 1109.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 1056.97 | 1089.67 | 1099.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 1038.80 | 1038.08 | 1053.64 | SL hit (close>ema200) qty=0.50 sl=1038.08 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1055.30 | 1046.47 | 1046.22 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1027.70 | 1045.12 | 1046.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1022.00 | 1032.27 | 1038.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 1032.90 | 1030.76 | 1036.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 11:00:00 | 1032.90 | 1030.76 | 1036.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1030.00 | 1030.60 | 1035.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 1033.50 | 1030.60 | 1035.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1008.10 | 1007.14 | 1015.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 1001.50 | 1005.71 | 1014.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1038.60 | 1015.32 | 1015.86 | SL hit (close>static) qty=1.00 sl=1022.30 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1038.40 | 1019.94 | 1017.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 1046.70 | 1025.29 | 1020.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1019.30 | 1030.72 | 1025.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1019.30 | 1030.72 | 1025.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1019.30 | 1030.72 | 1025.95 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1007.20 | 1021.31 | 1022.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 995.50 | 1014.26 | 1019.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 999.70 | 997.27 | 1006.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 999.70 | 997.27 | 1006.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 999.70 | 997.27 | 1006.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 994.85 | 997.04 | 1005.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:15:00 | 992.80 | 997.04 | 1005.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 994.40 | 998.76 | 1004.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:45:00 | 994.20 | 998.22 | 1003.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 994.75 | 989.11 | 995.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:30:00 | 994.60 | 989.11 | 995.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 995.95 | 990.48 | 995.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:45:00 | 994.00 | 990.48 | 995.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 994.75 | 991.33 | 995.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 992.65 | 991.33 | 995.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1004.70 | 994.01 | 996.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 1004.70 | 994.01 | 996.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 999.55 | 995.11 | 996.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:30:00 | 1005.65 | 995.11 | 996.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 997.45 | 995.36 | 996.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:00:00 | 997.45 | 995.36 | 996.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 1004.40 | 997.17 | 997.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:00:00 | 1004.40 | 997.17 | 997.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 1003.30 | 998.40 | 997.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 1003.30 | 998.40 | 997.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 1022.25 | 1004.10 | 1000.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1066.35 | 1068.51 | 1053.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:30:00 | 1066.90 | 1068.51 | 1053.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1068.20 | 1066.03 | 1056.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 1074.10 | 1066.03 | 1056.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 13:15:00 | 1077.10 | 1070.47 | 1060.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1078.80 | 1064.48 | 1063.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 1102.80 | 1122.95 | 1124.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1102.80 | 1122.95 | 1124.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1092.90 | 1116.94 | 1121.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 11:15:00 | 1092.25 | 1092.01 | 1099.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:00:00 | 1092.25 | 1092.01 | 1099.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1092.15 | 1091.74 | 1097.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:30:00 | 1099.00 | 1091.74 | 1097.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1103.80 | 1094.66 | 1098.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:00:00 | 1087.00 | 1096.01 | 1097.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 13:15:00 | 1085.80 | 1081.19 | 1080.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 1085.80 | 1081.19 | 1080.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 1098.50 | 1084.65 | 1082.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 1087.60 | 1089.61 | 1085.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 10:45:00 | 1090.60 | 1089.61 | 1085.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1087.80 | 1089.10 | 1086.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:00:00 | 1087.80 | 1089.10 | 1086.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 1045.10 | 1082.70 | 1084.44 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 12:15:00 | 1278.10 | 2025-05-20 12:15:00 | 1271.90 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-05-14 15:00:00 | 1284.20 | 2025-05-20 12:15:00 | 1271.90 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-05-15 09:30:00 | 1276.80 | 2025-05-20 12:15:00 | 1271.90 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-05-15 10:45:00 | 1278.50 | 2025-05-20 12:15:00 | 1271.90 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-05-21 14:30:00 | 1278.80 | 2025-05-23 15:15:00 | 1278.80 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-06-10 14:15:00 | 1211.80 | 2025-06-11 09:15:00 | 1226.50 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-06-17 11:15:00 | 1183.90 | 2025-06-17 14:15:00 | 1197.80 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-06-18 11:15:00 | 1184.20 | 2025-06-18 13:15:00 | 1201.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-06-24 15:00:00 | 1173.60 | 2025-06-25 13:15:00 | 1185.10 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-06-25 10:00:00 | 1171.80 | 2025-06-25 13:15:00 | 1185.10 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-07-01 10:30:00 | 1170.30 | 2025-07-03 11:15:00 | 1178.10 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-01 11:45:00 | 1168.40 | 2025-07-03 11:15:00 | 1178.10 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-01 13:45:00 | 1169.40 | 2025-07-03 11:15:00 | 1178.10 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest1 | 2025-07-11 09:15:00 | 1297.60 | 2025-07-14 09:15:00 | 1273.60 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-07-11 14:30:00 | 1287.90 | 2025-07-14 09:15:00 | 1273.60 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-11 15:15:00 | 1288.20 | 2025-07-14 09:15:00 | 1273.60 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-16 09:15:00 | 1265.80 | 2025-07-17 09:15:00 | 1278.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-23 09:15:00 | 1236.50 | 2025-07-31 09:15:00 | 1234.90 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-08-13 11:45:00 | 1187.40 | 2025-08-18 09:15:00 | 1205.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-08-14 10:30:00 | 1189.10 | 2025-08-18 09:15:00 | 1205.50 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-08-14 11:15:00 | 1186.00 | 2025-08-18 09:15:00 | 1205.50 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-08-19 14:30:00 | 1215.60 | 2025-08-29 09:15:00 | 1236.70 | STOP_HIT | 1.00 | 1.74% |
| BUY | retest2 | 2025-09-12 11:45:00 | 1245.60 | 2025-09-12 14:15:00 | 1248.30 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-09-12 12:45:00 | 1245.20 | 2025-09-12 14:15:00 | 1248.30 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-09-18 09:15:00 | 1233.70 | 2025-09-22 11:15:00 | 1242.90 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest1 | 2025-09-24 14:15:00 | 1191.00 | 2025-09-29 15:15:00 | 1178.00 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2025-09-30 10:30:00 | 1171.50 | 2025-10-13 11:15:00 | 1112.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 13:00:00 | 1171.60 | 2025-10-13 11:15:00 | 1113.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 14:45:00 | 1170.20 | 2025-10-13 11:15:00 | 1111.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 10:30:00 | 1171.50 | 2025-10-15 09:15:00 | 1112.90 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 13:00:00 | 1171.60 | 2025-10-15 09:15:00 | 1112.90 | STOP_HIT | 0.50 | 5.01% |
| SELL | retest2 | 2025-09-30 14:45:00 | 1170.20 | 2025-10-15 09:15:00 | 1112.90 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2025-11-10 13:15:00 | 1130.20 | 2025-11-11 14:15:00 | 1139.10 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-11-11 10:15:00 | 1130.20 | 2025-11-11 14:15:00 | 1139.10 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-11-11 11:15:00 | 1130.70 | 2025-11-11 14:15:00 | 1139.10 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-11-12 15:15:00 | 1139.00 | 2025-11-13 09:15:00 | 1127.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-19 13:15:00 | 1142.00 | 2025-11-25 13:15:00 | 1142.80 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-12-03 09:15:00 | 1120.40 | 2025-12-05 13:15:00 | 1133.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-05 09:45:00 | 1125.20 | 2025-12-05 13:15:00 | 1133.60 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest1 | 2025-12-09 09:15:00 | 1111.20 | 2025-12-09 10:15:00 | 1131.60 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-12-11 09:15:00 | 1135.10 | 2026-01-01 14:15:00 | 1244.98 | TARGET_HIT | 1.00 | 9.68% |
| BUY | retest2 | 2025-12-11 09:45:00 | 1131.80 | 2026-01-06 13:15:00 | 1248.61 | TARGET_HIT | 1.00 | 10.32% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1231.90 | 2026-01-16 09:15:00 | 1245.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-01-14 12:15:00 | 1231.30 | 2026-01-16 09:15:00 | 1245.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-14 13:00:00 | 1230.50 | 2026-01-16 09:15:00 | 1245.60 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-14 14:00:00 | 1230.70 | 2026-01-16 09:15:00 | 1245.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-22 11:15:00 | 1224.00 | 2026-01-22 13:15:00 | 1240.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-02-12 11:30:00 | 1209.90 | 2026-02-13 09:15:00 | 1192.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-03-10 12:15:00 | 1112.60 | 2026-03-12 09:15:00 | 1056.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 12:15:00 | 1112.60 | 2026-03-16 10:15:00 | 1038.80 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2026-03-24 10:30:00 | 1001.50 | 2026-03-25 09:15:00 | 1038.60 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-04-01 10:45:00 | 994.85 | 2026-04-06 14:15:00 | 1003.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-04-01 11:15:00 | 992.80 | 2026-04-06 14:15:00 | 1003.30 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-04-01 13:30:00 | 994.40 | 2026-04-06 14:15:00 | 1003.30 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-04-01 14:45:00 | 994.20 | 2026-04-06 14:15:00 | 1003.30 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-04-10 10:15:00 | 1074.10 | 2026-04-24 11:15:00 | 1102.80 | STOP_HIT | 1.00 | 2.67% |
| BUY | retest2 | 2026-04-10 13:15:00 | 1077.10 | 2026-04-24 11:15:00 | 1102.80 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1078.80 | 2026-04-24 11:15:00 | 1102.80 | STOP_HIT | 1.00 | 2.22% |
| SELL | retest2 | 2026-04-29 14:00:00 | 1087.00 | 2026-05-05 13:15:00 | 1085.80 | STOP_HIT | 1.00 | 0.11% |
