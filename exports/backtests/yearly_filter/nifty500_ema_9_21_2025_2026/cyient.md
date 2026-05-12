# Cyient Ltd. (CYIENT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 902.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 51 |
| ALERT2 | 49 |
| ALERT2_SKIP | 22 |
| ALERT3 | 128 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 55 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 44
- **Target hits / Stop hits / Partials:** 1 / 58 / 8
- **Avg / median % per leg:** 0.52% / -0.89%
- **Sum % (uncompounded):** 34.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 3 | 12.0% | 1 | 24 | 0 | -0.52% | -12.9% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.69% | -3.4% |
| BUY @ 3rd Alert (retest2) | 23 | 3 | 13.0% | 1 | 22 | 0 | -0.41% | -9.5% |
| SELL (all) | 42 | 20 | 47.6% | 0 | 34 | 8 | 1.14% | 47.9% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 4.61% | 13.8% |
| SELL @ 3rd Alert (retest2) | 39 | 17 | 43.6% | 0 | 32 | 7 | 0.87% | 34.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 4 | 1 | 2.09% | 10.4% |
| retest2 (combined) | 62 | 20 | 32.3% | 1 | 54 | 7 | 0.40% | 24.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1221.70 | 1187.71 | 1184.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 1230.10 | 1196.18 | 1188.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1240.80 | 1241.29 | 1225.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 1240.80 | 1241.29 | 1225.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1299.00 | 1302.17 | 1291.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 1290.20 | 1302.17 | 1291.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 1293.80 | 1299.00 | 1292.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 1293.80 | 1299.00 | 1292.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1299.10 | 1299.02 | 1293.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 1310.10 | 1298.81 | 1293.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 1291.90 | 1300.10 | 1296.97 | SL hit (close<static) qty=1.00 sl=1293.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 1295.40 | 1299.27 | 1299.35 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1315.80 | 1302.16 | 1300.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 14:15:00 | 1332.40 | 1318.90 | 1310.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 11:15:00 | 1352.90 | 1352.98 | 1339.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 11:45:00 | 1354.30 | 1352.98 | 1339.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1347.50 | 1349.82 | 1341.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1350.40 | 1349.03 | 1341.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 1351.00 | 1351.26 | 1344.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 1360.80 | 1349.13 | 1345.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 12:45:00 | 1350.30 | 1350.74 | 1347.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1349.60 | 1350.51 | 1348.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 1351.20 | 1350.51 | 1348.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1355.00 | 1351.41 | 1348.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:30:00 | 1350.30 | 1351.41 | 1348.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1343.00 | 1350.57 | 1348.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:15:00 | 1338.50 | 1350.57 | 1348.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1337.30 | 1347.92 | 1347.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 1337.30 | 1347.92 | 1347.79 | SL hit (close<static) qty=1.00 sl=1338.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 1344.10 | 1347.15 | 1347.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1336.80 | 1343.71 | 1345.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 10:15:00 | 1338.90 | 1337.38 | 1340.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 10:15:00 | 1338.90 | 1337.38 | 1340.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 1338.90 | 1337.38 | 1340.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:45:00 | 1343.00 | 1337.38 | 1340.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1328.40 | 1335.58 | 1339.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:15:00 | 1326.00 | 1335.58 | 1339.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 13:00:00 | 1325.30 | 1333.53 | 1338.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:15:00 | 1326.00 | 1331.38 | 1336.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:30:00 | 1327.10 | 1331.78 | 1335.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1345.20 | 1334.47 | 1336.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-04 10:15:00 | 1345.20 | 1334.47 | 1336.39 | SL hit (close>static) qty=1.00 sl=1339.40 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 1346.90 | 1338.93 | 1338.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 13:15:00 | 1354.80 | 1342.10 | 1339.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 15:15:00 | 1343.10 | 1343.42 | 1340.80 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:15:00 | 1354.00 | 1343.42 | 1340.80 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1332.80 | 1341.37 | 1340.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 1332.80 | 1341.37 | 1340.33 | SL hit (close<ema400) qty=1.00 sl=1340.33 alert=retest1 |

### Cycle 6 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1337.50 | 1339.70 | 1339.73 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 15:15:00 | 1341.00 | 1339.83 | 1339.78 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 1332.70 | 1338.41 | 1339.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 10:15:00 | 1328.70 | 1336.46 | 1338.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 1329.50 | 1329.34 | 1333.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 1329.50 | 1329.34 | 1333.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1329.50 | 1329.34 | 1333.02 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 1344.20 | 1333.90 | 1332.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 11:15:00 | 1361.00 | 1339.32 | 1335.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 1341.60 | 1344.57 | 1340.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1341.60 | 1344.57 | 1340.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1341.60 | 1344.57 | 1340.10 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1337.10 | 1339.87 | 1340.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1317.10 | 1335.31 | 1338.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 1306.70 | 1306.49 | 1315.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:30:00 | 1302.60 | 1306.49 | 1315.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1318.70 | 1309.65 | 1315.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1318.70 | 1309.65 | 1315.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1318.00 | 1311.32 | 1315.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1317.60 | 1311.32 | 1315.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1311.30 | 1311.32 | 1315.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1316.40 | 1311.32 | 1315.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1337.60 | 1316.57 | 1317.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 1337.60 | 1316.57 | 1317.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 1345.50 | 1322.36 | 1319.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 14:15:00 | 1360.30 | 1341.32 | 1333.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 1295.00 | 1334.25 | 1331.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 1295.00 | 1334.25 | 1331.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1295.00 | 1334.25 | 1331.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 1295.00 | 1334.25 | 1331.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 1290.00 | 1325.40 | 1327.81 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1317.50 | 1308.32 | 1308.04 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 12:15:00 | 1299.90 | 1309.54 | 1309.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 1283.70 | 1299.79 | 1304.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 1297.60 | 1292.48 | 1298.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 1297.60 | 1292.48 | 1298.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1299.80 | 1293.94 | 1298.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 1298.10 | 1293.94 | 1298.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1320.20 | 1299.19 | 1300.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 1320.20 | 1299.19 | 1300.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1309.10 | 1301.17 | 1301.28 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 1312.30 | 1303.40 | 1302.28 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1290.00 | 1301.01 | 1301.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 11:15:00 | 1285.40 | 1295.98 | 1299.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 13:15:00 | 1288.40 | 1287.59 | 1291.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 13:15:00 | 1288.40 | 1287.59 | 1291.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1288.40 | 1287.59 | 1291.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 1288.40 | 1287.59 | 1291.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 1293.70 | 1288.81 | 1291.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 1293.70 | 1288.81 | 1291.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1292.10 | 1289.47 | 1291.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1304.00 | 1289.47 | 1291.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 1296.90 | 1291.98 | 1292.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 1296.90 | 1291.98 | 1292.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 12:15:00 | 1299.40 | 1293.46 | 1293.17 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 1291.90 | 1295.81 | 1295.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 1283.00 | 1291.14 | 1293.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 14:15:00 | 1297.40 | 1291.43 | 1292.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 14:15:00 | 1297.40 | 1291.43 | 1292.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1297.40 | 1291.43 | 1292.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1297.40 | 1291.43 | 1292.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1298.30 | 1292.80 | 1293.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1302.80 | 1292.80 | 1293.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 1299.90 | 1294.22 | 1293.92 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 1291.20 | 1293.63 | 1293.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 15:15:00 | 1288.40 | 1292.13 | 1293.01 | Break + close below crossover candle low |

### Cycle 21 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 1307.80 | 1295.27 | 1294.36 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 1286.70 | 1293.98 | 1294.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 1279.80 | 1289.68 | 1292.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 14:15:00 | 1292.50 | 1288.40 | 1291.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 14:15:00 | 1292.50 | 1288.40 | 1291.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1292.50 | 1288.40 | 1291.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 1292.50 | 1288.40 | 1291.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1296.80 | 1290.08 | 1291.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 1285.10 | 1290.08 | 1291.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 1287.90 | 1289.50 | 1291.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 1293.50 | 1286.24 | 1285.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 1293.50 | 1286.24 | 1285.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 15:15:00 | 1296.10 | 1288.21 | 1286.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 1304.60 | 1307.07 | 1302.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 15:00:00 | 1304.60 | 1307.07 | 1302.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1308.20 | 1307.30 | 1302.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 1299.80 | 1307.30 | 1302.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1301.10 | 1306.06 | 1302.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 1298.00 | 1306.06 | 1302.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1302.80 | 1305.41 | 1302.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:15:00 | 1300.80 | 1305.41 | 1302.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1298.60 | 1304.05 | 1302.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 1298.60 | 1304.05 | 1302.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1296.10 | 1302.46 | 1301.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:30:00 | 1296.00 | 1302.46 | 1301.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 1296.40 | 1300.17 | 1300.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1283.40 | 1296.82 | 1299.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 1284.30 | 1281.09 | 1286.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 1284.30 | 1281.09 | 1286.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1284.30 | 1281.73 | 1286.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:30:00 | 1287.40 | 1281.73 | 1286.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1282.80 | 1282.20 | 1285.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 1288.80 | 1282.20 | 1285.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1279.50 | 1275.14 | 1279.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 1279.50 | 1275.14 | 1279.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1281.60 | 1276.43 | 1279.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 1281.60 | 1276.43 | 1279.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 1278.20 | 1276.79 | 1279.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:30:00 | 1285.80 | 1276.79 | 1279.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 1274.00 | 1276.23 | 1278.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:30:00 | 1261.30 | 1273.78 | 1277.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:00:00 | 1251.20 | 1253.55 | 1263.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 12:15:00 | 1198.23 | 1210.20 | 1217.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 14:15:00 | 1188.64 | 1201.88 | 1212.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 11:15:00 | 1198.40 | 1196.50 | 1205.84 | SL hit (close>ema200) qty=0.50 sl=1196.50 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 1210.00 | 1207.58 | 1207.52 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 1189.40 | 1203.95 | 1205.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1177.70 | 1194.92 | 1199.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1187.90 | 1185.11 | 1192.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 1186.90 | 1185.11 | 1192.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1192.60 | 1186.61 | 1192.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1183.30 | 1186.61 | 1192.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:45:00 | 1179.90 | 1186.18 | 1191.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 1182.90 | 1186.18 | 1191.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 13:15:00 | 1178.50 | 1173.26 | 1173.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 1178.50 | 1173.26 | 1173.21 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 1169.60 | 1172.53 | 1172.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 15:15:00 | 1168.00 | 1171.62 | 1172.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 1172.00 | 1171.70 | 1172.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 1172.00 | 1171.70 | 1172.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1172.00 | 1171.70 | 1172.40 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 12:15:00 | 1176.30 | 1173.02 | 1172.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 13:15:00 | 1179.90 | 1174.40 | 1173.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 1173.60 | 1174.24 | 1173.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 14:15:00 | 1173.60 | 1174.24 | 1173.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1173.60 | 1174.24 | 1173.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 1175.70 | 1174.24 | 1173.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1171.90 | 1173.77 | 1173.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1174.90 | 1173.77 | 1173.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 12:15:00 | 1177.50 | 1175.18 | 1174.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 1213.80 | 1233.32 | 1235.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 1213.80 | 1233.32 | 1235.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 1208.00 | 1228.26 | 1232.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1188.30 | 1188.15 | 1202.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:30:00 | 1189.30 | 1188.15 | 1202.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1183.40 | 1179.04 | 1191.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1188.30 | 1179.04 | 1191.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1191.10 | 1183.64 | 1190.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1191.10 | 1183.64 | 1190.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1193.60 | 1185.63 | 1190.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:15:00 | 1191.30 | 1185.63 | 1190.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1190.00 | 1186.50 | 1190.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 1190.00 | 1186.50 | 1190.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1189.00 | 1187.00 | 1190.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1199.70 | 1187.00 | 1190.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1204.00 | 1190.40 | 1191.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1209.00 | 1190.40 | 1191.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1204.20 | 1193.16 | 1192.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 15:15:00 | 1211.00 | 1201.54 | 1198.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 1196.90 | 1200.61 | 1198.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 1196.90 | 1200.61 | 1198.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1196.90 | 1200.61 | 1198.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 1200.60 | 1200.61 | 1198.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1192.60 | 1199.01 | 1197.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 1189.50 | 1199.01 | 1197.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 1189.90 | 1197.19 | 1197.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1181.00 | 1191.42 | 1194.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 1172.50 | 1172.19 | 1180.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 1174.40 | 1172.19 | 1180.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1177.40 | 1173.23 | 1180.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 1164.70 | 1174.19 | 1178.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1241.00 | 1191.08 | 1184.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1241.00 | 1191.08 | 1184.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 1257.30 | 1235.07 | 1228.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 1257.90 | 1259.20 | 1248.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 1257.90 | 1259.20 | 1248.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1252.80 | 1258.28 | 1252.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 1253.70 | 1258.28 | 1252.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1252.00 | 1257.02 | 1252.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1252.00 | 1257.02 | 1252.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1247.20 | 1255.06 | 1251.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 1243.50 | 1255.06 | 1251.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1248.40 | 1253.73 | 1251.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:45:00 | 1248.10 | 1253.73 | 1251.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 1242.80 | 1249.54 | 1250.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 1240.00 | 1247.63 | 1249.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 1158.40 | 1151.89 | 1161.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 13:15:00 | 1158.40 | 1151.89 | 1161.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1158.40 | 1151.89 | 1161.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 1158.40 | 1151.89 | 1161.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1164.40 | 1153.11 | 1159.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 1164.40 | 1153.11 | 1159.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1167.30 | 1155.94 | 1160.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 1167.30 | 1155.94 | 1160.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 1158.70 | 1156.50 | 1160.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:45:00 | 1157.90 | 1155.94 | 1159.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 1153.10 | 1152.22 | 1156.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1168.20 | 1156.88 | 1155.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 1168.20 | 1156.88 | 1155.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 1174.70 | 1162.54 | 1158.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 1179.70 | 1180.00 | 1173.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:30:00 | 1180.00 | 1180.00 | 1173.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1173.60 | 1179.75 | 1175.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1173.60 | 1179.75 | 1175.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1175.00 | 1178.80 | 1175.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 1173.40 | 1178.80 | 1175.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1178.40 | 1178.72 | 1175.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 1176.50 | 1178.72 | 1175.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1173.40 | 1177.65 | 1175.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 1173.40 | 1177.65 | 1175.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 1169.40 | 1176.00 | 1174.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 1170.80 | 1176.00 | 1174.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1176.00 | 1176.00 | 1174.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 1181.80 | 1176.00 | 1174.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 1178.00 | 1178.84 | 1177.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:30:00 | 1179.40 | 1179.31 | 1177.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 1165.50 | 1176.16 | 1177.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1165.50 | 1176.16 | 1177.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 1152.50 | 1168.92 | 1173.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 1165.00 | 1164.74 | 1170.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 15:00:00 | 1165.00 | 1164.74 | 1170.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1161.70 | 1163.25 | 1168.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 1158.20 | 1161.86 | 1167.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 13:45:00 | 1159.00 | 1158.92 | 1161.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 14:15:00 | 1183.50 | 1142.29 | 1147.22 | SL hit (close>static) qty=1.00 sl=1172.90 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1171.00 | 1152.15 | 1151.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 1185.30 | 1171.01 | 1162.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 1195.10 | 1205.39 | 1195.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 1195.10 | 1205.39 | 1195.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1195.10 | 1205.39 | 1195.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 1208.90 | 1197.27 | 1194.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:30:00 | 1213.00 | 1201.16 | 1196.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 15:00:00 | 1208.20 | 1204.17 | 1199.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 1184.70 | 1196.73 | 1197.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 1184.70 | 1196.73 | 1197.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 1181.70 | 1187.41 | 1190.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 1186.90 | 1181.18 | 1184.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 1186.90 | 1181.18 | 1184.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1186.90 | 1181.18 | 1184.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:30:00 | 1191.60 | 1181.18 | 1184.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1181.20 | 1181.18 | 1184.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 1178.10 | 1181.17 | 1184.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1119.19 | 1141.42 | 1150.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 1123.90 | 1120.17 | 1132.58 | SL hit (close>ema200) qty=0.50 sl=1120.17 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 1143.00 | 1137.53 | 1137.48 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1131.40 | 1136.31 | 1136.93 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 1147.50 | 1137.95 | 1137.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 1152.20 | 1140.80 | 1138.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 1162.90 | 1164.38 | 1157.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 1162.90 | 1164.38 | 1157.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1156.00 | 1162.39 | 1157.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 1156.00 | 1162.39 | 1157.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1152.90 | 1160.49 | 1157.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1152.90 | 1160.49 | 1157.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1152.00 | 1158.79 | 1156.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1144.60 | 1158.79 | 1156.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1143.90 | 1152.89 | 1154.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1128.00 | 1145.37 | 1150.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 1134.80 | 1132.94 | 1139.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 14:45:00 | 1134.40 | 1132.94 | 1139.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1116.90 | 1129.48 | 1136.45 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 1154.50 | 1137.75 | 1136.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 15:15:00 | 1155.00 | 1147.13 | 1141.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 1150.70 | 1152.57 | 1147.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 1150.70 | 1152.57 | 1147.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1150.70 | 1152.57 | 1147.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 1150.70 | 1152.57 | 1147.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 1148.00 | 1151.65 | 1147.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 1137.90 | 1151.65 | 1147.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1131.30 | 1147.58 | 1146.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 1131.40 | 1147.58 | 1146.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1131.50 | 1144.37 | 1144.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1121.60 | 1134.54 | 1139.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1116.30 | 1111.36 | 1118.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1116.30 | 1111.36 | 1118.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1116.30 | 1111.36 | 1118.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1116.30 | 1111.36 | 1118.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1116.00 | 1112.29 | 1118.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 1118.00 | 1112.29 | 1118.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1117.30 | 1113.29 | 1118.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:15:00 | 1113.50 | 1113.29 | 1118.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:00:00 | 1115.40 | 1113.20 | 1117.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:45:00 | 1115.10 | 1114.22 | 1117.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1119.50 | 1115.88 | 1117.74 | SL hit (close>static) qty=1.00 sl=1119.20 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 12:15:00 | 1123.00 | 1117.23 | 1116.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 1137.80 | 1123.82 | 1120.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 10:15:00 | 1170.20 | 1170.45 | 1157.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:45:00 | 1168.70 | 1170.45 | 1157.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1175.80 | 1179.40 | 1173.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:45:00 | 1184.10 | 1180.34 | 1174.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 1170.20 | 1176.15 | 1174.48 | SL hit (close<static) qty=1.00 sl=1171.40 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1159.40 | 1171.58 | 1172.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 1151.60 | 1167.58 | 1170.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 11:15:00 | 1140.70 | 1138.18 | 1145.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 12:00:00 | 1140.70 | 1138.18 | 1145.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1152.00 | 1141.46 | 1144.46 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1154.90 | 1145.87 | 1145.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 1160.00 | 1150.54 | 1147.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 12:15:00 | 1147.20 | 1149.88 | 1147.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 12:15:00 | 1147.20 | 1149.88 | 1147.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1147.20 | 1149.88 | 1147.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:45:00 | 1146.60 | 1149.88 | 1147.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1168.00 | 1153.50 | 1149.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:30:00 | 1161.50 | 1153.50 | 1149.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1156.30 | 1157.66 | 1153.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:45:00 | 1158.10 | 1157.66 | 1153.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1155.90 | 1157.31 | 1153.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 1155.30 | 1157.31 | 1153.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 1155.90 | 1156.66 | 1153.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:30:00 | 1155.00 | 1156.66 | 1153.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1158.60 | 1157.05 | 1154.35 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1150.00 | 1152.95 | 1153.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 1145.80 | 1151.52 | 1152.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 1143.60 | 1142.27 | 1146.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 15:00:00 | 1143.60 | 1142.27 | 1146.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1142.30 | 1141.27 | 1144.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:30:00 | 1131.50 | 1140.88 | 1143.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 1151.10 | 1144.36 | 1144.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 1151.10 | 1144.36 | 1144.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1158.70 | 1147.23 | 1145.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1156.70 | 1156.71 | 1152.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1156.70 | 1156.71 | 1152.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1156.70 | 1156.71 | 1152.63 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 14:15:00 | 1146.70 | 1151.05 | 1151.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 1133.30 | 1146.29 | 1148.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 1113.10 | 1111.80 | 1120.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 1113.10 | 1111.80 | 1120.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1101.50 | 1095.98 | 1105.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1115.70 | 1095.98 | 1105.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1122.10 | 1101.21 | 1106.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 1122.10 | 1101.21 | 1106.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1117.60 | 1104.49 | 1107.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 1122.50 | 1104.49 | 1107.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1121.00 | 1110.09 | 1109.85 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 09:15:00 | 1105.30 | 1110.63 | 1111.24 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1136.00 | 1114.42 | 1112.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 1157.40 | 1127.01 | 1118.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 1145.00 | 1145.09 | 1134.53 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 12:45:00 | 1153.20 | 1146.54 | 1136.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 1132.20 | 1143.17 | 1137.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 15:15:00 | 1132.20 | 1143.17 | 1137.15 | SL hit (close<ema400) qty=1.00 sl=1137.15 alert=retest1 |

### Cycle 54 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 1144.20 | 1163.70 | 1165.29 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 1170.70 | 1163.71 | 1162.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1190.00 | 1174.40 | 1170.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 10:15:00 | 1193.90 | 1195.31 | 1186.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 11:00:00 | 1193.90 | 1195.31 | 1186.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 1185.70 | 1193.51 | 1186.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:00:00 | 1185.70 | 1193.51 | 1186.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 1183.00 | 1191.41 | 1186.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 1183.00 | 1191.41 | 1186.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 1181.80 | 1189.49 | 1186.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 1181.80 | 1189.49 | 1186.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1182.00 | 1187.99 | 1185.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1177.40 | 1187.99 | 1185.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 1173.20 | 1182.76 | 1183.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 1135.00 | 1173.49 | 1179.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 14:15:00 | 1133.10 | 1131.08 | 1144.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 15:00:00 | 1133.10 | 1131.08 | 1144.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1115.00 | 1103.83 | 1111.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:30:00 | 1105.70 | 1103.74 | 1110.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 1104.20 | 1111.61 | 1112.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1104.30 | 1108.29 | 1109.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 1120.90 | 1112.12 | 1111.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 1120.90 | 1112.12 | 1111.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 1131.00 | 1117.27 | 1113.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 1120.00 | 1133.71 | 1127.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 1120.00 | 1133.71 | 1127.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1120.00 | 1133.71 | 1127.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:15:00 | 1114.50 | 1133.71 | 1127.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 1109.40 | 1128.85 | 1125.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 1109.40 | 1128.85 | 1125.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1130.20 | 1125.90 | 1124.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1130.20 | 1125.90 | 1124.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1136.50 | 1128.02 | 1125.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 1151.80 | 1128.02 | 1125.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1122.70 | 1141.61 | 1136.62 | SL hit (close<static) qty=1.00 sl=1125.30 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 11:15:00 | 1120.10 | 1133.34 | 1133.48 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 1140.90 | 1134.60 | 1134.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 1147.00 | 1137.08 | 1135.20 | Break + close above crossover candle high |

### Cycle 60 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 1113.10 | 1134.35 | 1134.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 1102.10 | 1119.25 | 1126.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 1082.20 | 1073.63 | 1090.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 1087.00 | 1080.82 | 1086.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1087.00 | 1080.82 | 1086.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:45:00 | 1088.30 | 1080.82 | 1086.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1089.90 | 1082.64 | 1087.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:45:00 | 1089.10 | 1082.64 | 1087.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1092.40 | 1084.59 | 1087.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:45:00 | 1092.90 | 1084.59 | 1087.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1089.50 | 1085.57 | 1087.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:45:00 | 1092.20 | 1085.57 | 1087.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1079.80 | 1084.42 | 1087.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:15:00 | 1066.00 | 1080.44 | 1084.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 15:15:00 | 1012.70 | 1031.74 | 1050.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 1010.60 | 997.97 | 1010.82 | SL hit (close>ema200) qty=0.50 sl=997.97 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 1016.00 | 1012.11 | 1012.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 15:15:00 | 1024.90 | 1015.50 | 1013.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 1013.90 | 1015.18 | 1013.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1013.90 | 1015.18 | 1013.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1013.90 | 1015.18 | 1013.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 1009.90 | 1015.18 | 1013.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1012.20 | 1014.58 | 1013.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 1011.70 | 1014.58 | 1013.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1008.80 | 1013.43 | 1013.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1008.80 | 1013.43 | 1013.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1006.00 | 1011.94 | 1012.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 992.00 | 1006.79 | 1009.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 15:15:00 | 988.70 | 984.24 | 990.92 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:15:00 | 962.10 | 984.24 | 990.92 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 920.90 | 926.98 | 939.37 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 14:15:00 | 914.00 | 923.51 | 932.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 913.50 | 923.51 | 932.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 10:00:00 | 916.80 | 920.33 | 929.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 884.05 | 883.79 | 894.58 | SL hit (close>ema200) qty=0.50 sl=883.79 alert=retest1 |

### Cycle 63 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 896.85 | 881.82 | 880.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 909.00 | 887.26 | 883.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 918.00 | 922.56 | 908.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:15:00 | 918.15 | 922.56 | 908.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 906.80 | 919.41 | 908.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 906.80 | 919.41 | 908.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 918.75 | 919.28 | 908.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 923.75 | 919.79 | 910.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 902.05 | 914.36 | 909.93 | SL hit (close<static) qty=1.00 sl=905.65 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 881.50 | 906.45 | 907.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 879.00 | 900.96 | 904.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 864.20 | 859.93 | 872.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:45:00 | 864.15 | 859.93 | 872.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 869.80 | 858.53 | 865.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 850.10 | 862.16 | 864.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 807.60 | 827.12 | 838.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 803.50 | 797.73 | 809.57 | SL hit (close>ema200) qty=0.50 sl=797.73 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 822.30 | 811.10 | 811.10 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 794.35 | 808.84 | 810.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 782.40 | 796.83 | 803.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 800.20 | 775.63 | 785.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 800.20 | 775.63 | 785.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 800.20 | 775.63 | 785.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 804.20 | 775.63 | 785.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 787.60 | 778.02 | 785.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 785.20 | 782.66 | 786.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 803.65 | 785.88 | 785.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 803.65 | 785.88 | 785.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 810.80 | 790.86 | 787.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 888.55 | 897.38 | 881.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 888.55 | 897.38 | 881.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 888.55 | 897.38 | 881.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 896.35 | 897.38 | 881.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 985.99 | 961.79 | 945.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 946.85 | 955.58 | 956.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 943.20 | 953.10 | 955.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 15:15:00 | 890.40 | 886.11 | 900.52 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:15:00 | 880.05 | 886.11 | 900.52 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 873.70 | 855.97 | 864.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-30 11:15:00 | 873.70 | 855.97 | 864.75 | SL hit (close>ema400) qty=1.00 sl=864.75 alert=retest1 |

### Cycle 69 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 871.65 | 867.61 | 867.44 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 865.00 | 867.42 | 867.45 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 871.00 | 867.93 | 867.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 874.85 | 869.54 | 868.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 887.40 | 888.69 | 881.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 12:00:00 | 887.40 | 888.69 | 881.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 09:15:00 | 1310.10 | 2025-05-20 13:15:00 | 1291.90 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-05-21 09:30:00 | 1301.50 | 2025-05-22 12:15:00 | 1295.40 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-05-21 12:15:00 | 1300.80 | 2025-05-22 12:15:00 | 1295.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-05-22 09:45:00 | 1299.40 | 2025-05-22 12:15:00 | 1295.40 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1350.40 | 2025-05-30 10:15:00 | 1337.30 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-05-28 10:45:00 | 1351.00 | 2025-05-30 10:15:00 | 1337.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1360.80 | 2025-05-30 10:15:00 | 1337.30 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-05-29 12:45:00 | 1350.30 | 2025-05-30 10:15:00 | 1337.30 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-03 12:15:00 | 1326.00 | 2025-06-04 10:15:00 | 1345.20 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-06-03 13:00:00 | 1325.30 | 2025-06-04 10:15:00 | 1345.20 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-06-03 15:15:00 | 1326.00 | 2025-06-04 10:15:00 | 1345.20 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-06-04 09:30:00 | 1327.10 | 2025-06-04 10:15:00 | 1345.20 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2025-06-05 09:15:00 | 1354.00 | 2025-06-05 10:15:00 | 1332.80 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-07-11 09:15:00 | 1285.10 | 2025-07-14 14:15:00 | 1293.50 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-07-11 09:45:00 | 1287.90 | 2025-07-14 14:15:00 | 1293.50 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-07-24 09:30:00 | 1261.30 | 2025-08-01 12:15:00 | 1198.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 10:00:00 | 1251.20 | 2025-08-01 14:15:00 | 1188.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 09:30:00 | 1261.30 | 2025-08-04 11:15:00 | 1198.40 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2025-07-25 10:00:00 | 1251.20 | 2025-08-04 11:15:00 | 1198.40 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1183.30 | 2025-08-13 13:15:00 | 1178.50 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-08-08 09:45:00 | 1179.90 | 2025-08-13 13:15:00 | 1178.50 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-08-08 10:15:00 | 1182.90 | 2025-08-13 13:15:00 | 1178.50 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1174.90 | 2025-08-26 12:15:00 | 1213.80 | STOP_HIT | 1.00 | 3.31% |
| BUY | retest2 | 2025-08-18 12:15:00 | 1177.50 | 2025-08-26 12:15:00 | 1213.80 | STOP_HIT | 1.00 | 3.08% |
| SELL | retest2 | 2025-09-08 15:00:00 | 1164.70 | 2025-09-10 09:15:00 | 1241.00 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2025-09-30 12:45:00 | 1157.90 | 2025-10-03 13:15:00 | 1168.20 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-10-01 09:45:00 | 1153.10 | 2025-10-03 13:15:00 | 1168.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-10-09 09:15:00 | 1181.80 | 2025-10-13 09:15:00 | 1165.50 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-10-09 14:00:00 | 1178.00 | 2025-10-13 09:15:00 | 1165.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-09 14:30:00 | 1179.40 | 2025-10-13 09:15:00 | 1165.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-14 10:30:00 | 1158.20 | 2025-10-16 14:15:00 | 1183.50 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-10-15 13:45:00 | 1159.00 | 2025-10-16 14:15:00 | 1183.50 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-10-27 10:45:00 | 1208.90 | 2025-10-28 13:15:00 | 1184.70 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-27 12:30:00 | 1213.00 | 2025-10-28 13:15:00 | 1184.70 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-10-27 15:00:00 | 1208.20 | 2025-10-28 13:15:00 | 1184.70 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-10-31 12:15:00 | 1178.10 | 2025-11-07 09:15:00 | 1119.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 12:15:00 | 1178.10 | 2025-11-10 09:15:00 | 1123.90 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2025-11-26 12:15:00 | 1113.50 | 2025-11-27 09:15:00 | 1119.50 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-26 14:00:00 | 1115.40 | 2025-11-27 09:15:00 | 1119.50 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-11-26 14:45:00 | 1115.10 | 2025-11-27 09:15:00 | 1119.50 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-11-27 09:30:00 | 1114.70 | 2025-11-28 11:15:00 | 1125.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-27 11:15:00 | 1113.70 | 2025-11-28 11:15:00 | 1125.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1112.20 | 2025-11-28 11:15:00 | 1125.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-05 10:45:00 | 1184.10 | 2025-12-05 14:15:00 | 1170.20 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-12-18 14:30:00 | 1131.50 | 2025-12-19 13:15:00 | 1151.10 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest1 | 2026-01-06 12:45:00 | 1153.20 | 2026-01-06 15:15:00 | 1132.20 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-01-07 09:15:00 | 1151.00 | 2026-01-12 09:15:00 | 1144.20 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1148.50 | 2026-01-12 09:15:00 | 1144.20 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2026-01-07 10:45:00 | 1164.30 | 2026-01-12 09:15:00 | 1144.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-01-28 09:30:00 | 1105.70 | 2026-01-30 10:15:00 | 1120.90 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-01-29 10:15:00 | 1104.20 | 2026-01-30 10:15:00 | 1120.90 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1104.30 | 2026-01-30 10:15:00 | 1120.90 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-03 09:15:00 | 1151.80 | 2026-02-04 09:15:00 | 1122.70 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-02-11 12:15:00 | 1066.00 | 2026-02-12 15:15:00 | 1012.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 12:15:00 | 1066.00 | 2026-02-17 09:15:00 | 1010.60 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest1 | 2026-02-24 09:15:00 | 962.10 | 2026-02-27 14:15:00 | 914.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-24 09:15:00 | 962.10 | 2026-03-05 12:15:00 | 884.05 | STOP_HIT | 0.50 | 8.11% |
| SELL | retest2 | 2026-02-27 15:00:00 | 913.50 | 2026-03-06 09:15:00 | 870.96 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2026-02-27 15:00:00 | 913.50 | 2026-03-06 09:15:00 | 890.00 | STOP_HIT | 0.50 | 2.57% |
| SELL | retest2 | 2026-03-02 10:00:00 | 916.80 | 2026-03-09 09:15:00 | 867.82 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2026-03-02 10:00:00 | 916.80 | 2026-03-10 09:15:00 | 881.55 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest2 | 2026-03-12 11:45:00 | 923.75 | 2026-03-12 14:15:00 | 902.05 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-03-19 09:15:00 | 850.10 | 2026-03-23 09:15:00 | 807.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 850.10 | 2026-03-24 13:15:00 | 803.50 | STOP_HIT | 0.50 | 5.48% |
| SELL | retest2 | 2026-04-01 13:15:00 | 785.20 | 2026-04-02 13:15:00 | 803.65 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-04-13 10:15:00 | 896.35 | 2026-04-17 09:15:00 | 985.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-04-28 09:15:00 | 880.05 | 2026-04-30 11:15:00 | 873.70 | STOP_HIT | 1.00 | 0.72% |
