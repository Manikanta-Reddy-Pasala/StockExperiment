# Voltas Ltd. (VOLTAS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1323.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 142 |
| ALERT1 | 96 |
| ALERT2 | 93 |
| ALERT2_SKIP | 42 |
| ALERT3 | 277 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 117 |
| PARTIAL | 20 |
| TARGET_HIT | 10 |
| STOP_HIT | 112 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 86
- **Target hits / Stop hits / Partials:** 10 / 112 / 20
- **Avg / median % per leg:** 0.93% / -0.72%
- **Sum % (uncompounded):** 132.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 11 | 21.2% | 5 | 47 | 0 | 0.05% | 2.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.49% | -1.5% |
| BUY @ 3rd Alert (retest2) | 51 | 11 | 21.6% | 5 | 46 | 0 | 0.08% | 4.0% |
| SELL (all) | 90 | 45 | 50.0% | 5 | 65 | 20 | 1.45% | 130.2% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.39% | -5.6% |
| SELL @ 3rd Alert (retest2) | 86 | 45 | 52.3% | 5 | 61 | 20 | 1.58% | 135.8% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.41% | -7.1% |
| retest2 (combined) | 137 | 56 | 40.9% | 10 | 107 | 20 | 1.02% | 139.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 09:30:00 | 1282.95 | 1289.41 | 1310.06 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 1299.80 | 1289.79 | 1301.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:45:00 | 1307.25 | 1289.79 | 1301.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 1301.85 | 1292.21 | 1301.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-13 15:15:00 | 1301.85 | 1292.21 | 1301.69 | SL hit (close>ema400) qty=1.00 sl=1301.69 alert=retest1 |

### Cycle 2 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 1325.70 | 1308.40 | 1306.67 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 1303.00 | 1305.96 | 1306.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 15:15:00 | 1300.00 | 1304.77 | 1305.73 | Break + close below crossover candle low |

### Cycle 4 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 1317.80 | 1307.37 | 1306.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 14:15:00 | 1320.45 | 1311.79 | 1309.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 1301.25 | 1311.47 | 1309.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 09:15:00 | 1301.25 | 1311.47 | 1309.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 1301.25 | 1311.47 | 1309.81 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 10:15:00 | 1288.05 | 1306.79 | 1307.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 12:15:00 | 1286.80 | 1300.10 | 1304.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 1297.50 | 1295.25 | 1300.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 11:15:00 | 1296.00 | 1295.40 | 1299.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 1296.00 | 1295.40 | 1299.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 1296.25 | 1295.40 | 1299.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1299.95 | 1296.74 | 1299.76 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 1325.75 | 1303.60 | 1300.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 1342.30 | 1311.34 | 1304.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 11:15:00 | 1386.90 | 1402.05 | 1383.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 12:00:00 | 1386.90 | 1402.05 | 1383.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 1396.65 | 1400.97 | 1385.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 13:15:00 | 1400.05 | 1400.97 | 1385.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 14:30:00 | 1398.40 | 1398.71 | 1386.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 10:30:00 | 1397.45 | 1396.24 | 1388.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 14:15:00 | 1376.30 | 1390.13 | 1387.96 | SL hit (close<static) qty=1.00 sl=1383.60 alert=retest2 |

### Cycle 7 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 1355.05 | 1380.29 | 1383.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 1354.00 | 1370.99 | 1378.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 1372.95 | 1361.70 | 1370.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 1372.95 | 1361.70 | 1370.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1372.95 | 1361.70 | 1370.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 1377.35 | 1361.70 | 1370.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1381.35 | 1365.63 | 1371.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:30:00 | 1382.20 | 1365.63 | 1371.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 1375.00 | 1367.50 | 1371.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:30:00 | 1371.40 | 1368.88 | 1371.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:15:00 | 1365.60 | 1370.01 | 1371.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1423.25 | 1377.58 | 1374.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1423.25 | 1377.58 | 1374.61 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1302.95 | 1372.18 | 1378.74 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 1450.05 | 1391.47 | 1384.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 1464.20 | 1421.67 | 1401.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 1441.50 | 1444.12 | 1422.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 12:00:00 | 1441.50 | 1444.12 | 1422.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1441.80 | 1446.13 | 1431.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 11:15:00 | 1462.00 | 1445.22 | 1439.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 12:00:00 | 1459.55 | 1448.08 | 1440.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 13:15:00 | 1458.90 | 1449.06 | 1442.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 14:45:00 | 1461.80 | 1453.43 | 1445.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1464.55 | 1456.23 | 1448.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:30:00 | 1471.10 | 1459.12 | 1450.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 13:15:00 | 1446.05 | 1456.22 | 1451.09 | SL hit (close<static) qty=1.00 sl=1448.05 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 14:15:00 | 1447.10 | 1450.79 | 1451.00 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 09:15:00 | 1473.25 | 1454.62 | 1452.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 10:15:00 | 1482.25 | 1460.15 | 1455.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 09:15:00 | 1475.15 | 1475.56 | 1466.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 10:00:00 | 1475.15 | 1475.56 | 1466.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1498.75 | 1519.95 | 1507.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 1504.90 | 1519.95 | 1507.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 1499.10 | 1515.78 | 1506.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:30:00 | 1499.05 | 1515.78 | 1506.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 1476.30 | 1501.05 | 1501.43 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 1523.40 | 1497.71 | 1495.58 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 13:15:00 | 1500.00 | 1504.42 | 1504.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 1495.00 | 1502.54 | 1503.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 1506.50 | 1502.29 | 1503.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 1506.50 | 1502.29 | 1503.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1506.50 | 1502.29 | 1503.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 1506.50 | 1502.29 | 1503.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 1503.00 | 1502.43 | 1503.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:30:00 | 1505.00 | 1502.43 | 1503.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1499.35 | 1501.81 | 1503.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:30:00 | 1494.05 | 1500.83 | 1502.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:30:00 | 1493.80 | 1499.92 | 1501.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:30:00 | 1492.50 | 1497.58 | 1500.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:15:00 | 1419.35 | 1441.54 | 1459.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:15:00 | 1419.11 | 1441.54 | 1459.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:15:00 | 1417.88 | 1441.54 | 1459.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-02 14:15:00 | 1450.65 | 1441.72 | 1456.71 | SL hit (close>ema200) qty=0.50 sl=1441.72 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 1469.00 | 1455.28 | 1453.60 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 1446.75 | 1451.86 | 1452.45 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 14:15:00 | 1459.00 | 1453.29 | 1453.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 15:15:00 | 1462.95 | 1455.22 | 1453.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 1459.50 | 1464.53 | 1460.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 1459.50 | 1464.53 | 1460.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1459.50 | 1464.53 | 1460.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 1459.50 | 1464.53 | 1460.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1464.55 | 1464.53 | 1460.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 1461.65 | 1464.53 | 1460.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1524.15 | 1527.40 | 1519.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:45:00 | 1521.45 | 1527.40 | 1519.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1521.55 | 1526.23 | 1519.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:45:00 | 1516.00 | 1526.23 | 1519.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 1523.30 | 1525.64 | 1520.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 1508.90 | 1525.64 | 1520.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1509.80 | 1522.48 | 1519.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1506.85 | 1522.48 | 1519.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 1494.00 | 1516.78 | 1517.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 1492.80 | 1505.55 | 1510.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 1492.85 | 1484.60 | 1493.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 1492.85 | 1484.60 | 1493.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1492.85 | 1484.60 | 1493.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 1503.45 | 1484.60 | 1493.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1490.50 | 1485.78 | 1493.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:45:00 | 1482.75 | 1484.61 | 1491.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 13:30:00 | 1478.65 | 1482.59 | 1490.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 1481.95 | 1483.09 | 1489.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:00:00 | 1481.40 | 1478.08 | 1483.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 1480.65 | 1478.59 | 1483.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 1503.45 | 1478.59 | 1483.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1498.30 | 1482.54 | 1484.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 1501.20 | 1482.54 | 1484.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-24 10:15:00 | 1511.00 | 1488.23 | 1487.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 1511.00 | 1488.23 | 1487.02 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 11:15:00 | 1472.00 | 1486.32 | 1488.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 12:15:00 | 1468.45 | 1482.74 | 1486.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 1486.15 | 1475.10 | 1480.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 1486.15 | 1475.10 | 1480.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 1486.15 | 1475.10 | 1480.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 1486.15 | 1475.10 | 1480.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 1492.80 | 1478.64 | 1481.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 1492.80 | 1478.64 | 1481.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 1480.00 | 1481.62 | 1482.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:30:00 | 1484.20 | 1481.62 | 1482.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 1490.00 | 1483.30 | 1483.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 15:00:00 | 1490.00 | 1483.30 | 1483.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 15:15:00 | 1494.65 | 1485.57 | 1484.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 1514.05 | 1491.27 | 1487.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 12:15:00 | 1538.55 | 1539.63 | 1528.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 13:00:00 | 1538.55 | 1539.63 | 1528.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1533.75 | 1538.42 | 1531.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 1533.75 | 1538.42 | 1531.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 1530.05 | 1536.74 | 1531.55 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 1500.60 | 1528.46 | 1529.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 15:15:00 | 1487.90 | 1502.81 | 1514.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1494.85 | 1466.38 | 1482.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1494.85 | 1466.38 | 1482.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1494.85 | 1466.38 | 1482.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 1494.85 | 1466.38 | 1482.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1477.10 | 1468.53 | 1482.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 1469.10 | 1470.79 | 1480.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 10:00:00 | 1469.95 | 1468.16 | 1476.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 13:15:00 | 1466.65 | 1470.88 | 1475.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 1456.15 | 1470.24 | 1474.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1463.40 | 1468.88 | 1473.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 1494.40 | 1475.16 | 1475.39 | SL hit (close>static) qty=1.00 sl=1494.20 alert=retest2 |

### Cycle 24 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 1548.40 | 1473.51 | 1464.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 12:15:00 | 1575.55 | 1507.49 | 1482.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 1571.60 | 1571.72 | 1540.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:30:00 | 1572.00 | 1571.72 | 1540.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 1549.80 | 1563.20 | 1548.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:00:00 | 1549.80 | 1563.20 | 1548.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 1548.75 | 1560.31 | 1548.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:30:00 | 1547.70 | 1560.31 | 1548.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 1549.95 | 1558.24 | 1548.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:45:00 | 1550.00 | 1558.24 | 1548.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 1550.65 | 1556.72 | 1548.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 1581.80 | 1556.72 | 1548.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 12:15:00 | 1531.90 | 1551.80 | 1548.97 | SL hit (close<static) qty=1.00 sl=1547.80 alert=retest2 |

### Cycle 25 — SELL (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 15:15:00 | 1532.15 | 1545.62 | 1546.75 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 1568.20 | 1550.13 | 1548.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 1608.35 | 1580.82 | 1565.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1675.65 | 1679.02 | 1661.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 1675.65 | 1679.02 | 1661.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1737.65 | 1763.29 | 1756.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 1737.65 | 1763.29 | 1756.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 1754.80 | 1761.59 | 1756.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 1764.60 | 1761.59 | 1756.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 11:15:00 | 1768.60 | 1761.52 | 1757.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 15:15:00 | 1790.80 | 1760.71 | 1758.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 15:15:00 | 1781.90 | 1783.87 | 1784.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 15:15:00 | 1781.90 | 1783.87 | 1784.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 1770.00 | 1781.09 | 1782.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 14:15:00 | 1778.90 | 1776.89 | 1779.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 14:15:00 | 1778.90 | 1776.89 | 1779.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1778.90 | 1776.89 | 1779.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 15:00:00 | 1778.90 | 1776.89 | 1779.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 1786.50 | 1778.81 | 1780.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 1770.70 | 1778.81 | 1780.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 1791.90 | 1781.43 | 1781.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 09:15:00 | 1791.90 | 1781.43 | 1781.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 12:15:00 | 1799.25 | 1787.09 | 1784.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 1827.35 | 1829.22 | 1817.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 15:00:00 | 1827.35 | 1829.22 | 1817.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1898.00 | 1902.47 | 1895.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:30:00 | 1899.90 | 1902.47 | 1895.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1897.30 | 1901.44 | 1895.94 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 1888.05 | 1892.89 | 1893.29 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 10:15:00 | 1902.75 | 1894.89 | 1894.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 13:15:00 | 1911.70 | 1898.99 | 1896.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 09:15:00 | 1897.25 | 1908.34 | 1902.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 1897.25 | 1908.34 | 1902.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1897.25 | 1908.34 | 1902.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 1902.80 | 1908.34 | 1902.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1928.75 | 1912.42 | 1904.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:15:00 | 1932.10 | 1912.42 | 1904.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 13:15:00 | 1887.55 | 1905.25 | 1906.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 13:15:00 | 1887.55 | 1905.25 | 1906.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 14:15:00 | 1883.35 | 1900.87 | 1904.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 09:15:00 | 1908.85 | 1899.85 | 1903.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 1908.85 | 1899.85 | 1903.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1908.85 | 1899.85 | 1903.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 1908.85 | 1899.85 | 1903.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 1907.50 | 1901.38 | 1903.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:15:00 | 1909.35 | 1901.38 | 1903.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 1905.70 | 1902.24 | 1903.80 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 13:15:00 | 1910.00 | 1904.95 | 1904.83 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 1888.25 | 1904.06 | 1905.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 10:15:00 | 1882.10 | 1899.67 | 1903.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 11:15:00 | 1870.10 | 1864.83 | 1878.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 12:00:00 | 1870.10 | 1864.83 | 1878.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 1857.20 | 1863.30 | 1876.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:00:00 | 1854.40 | 1861.52 | 1874.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 11:00:00 | 1854.80 | 1863.90 | 1872.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 11:30:00 | 1854.70 | 1860.91 | 1869.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:00:00 | 1854.85 | 1853.58 | 1862.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 1849.80 | 1853.39 | 1859.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:30:00 | 1847.85 | 1853.95 | 1859.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 15:15:00 | 1860.65 | 1855.29 | 1859.34 | SL hit (close>static) qty=1.00 sl=1860.35 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 1794.85 | 1786.66 | 1785.89 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 1779.20 | 1784.74 | 1785.12 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 13:15:00 | 1795.15 | 1786.82 | 1786.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 13:15:00 | 1802.55 | 1795.18 | 1791.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 1843.55 | 1859.75 | 1836.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 1843.55 | 1859.75 | 1836.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1843.55 | 1859.75 | 1836.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1843.55 | 1859.75 | 1836.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1842.85 | 1856.37 | 1837.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:30:00 | 1844.20 | 1856.37 | 1837.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 1845.80 | 1851.72 | 1838.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:30:00 | 1839.45 | 1851.72 | 1838.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 1853.60 | 1852.09 | 1839.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 14:15:00 | 1859.10 | 1852.09 | 1839.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 14:15:00 | 1835.00 | 1848.67 | 1839.42 | SL hit (close<static) qty=1.00 sl=1835.65 alert=retest2 |

### Cycle 37 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 1813.15 | 1841.55 | 1844.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 1791.20 | 1820.13 | 1831.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 11:15:00 | 1814.05 | 1809.54 | 1821.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:45:00 | 1817.00 | 1809.54 | 1821.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1820.85 | 1811.80 | 1821.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:30:00 | 1823.60 | 1811.80 | 1821.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 1816.50 | 1812.74 | 1821.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:15:00 | 1807.95 | 1812.74 | 1821.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 10:30:00 | 1807.10 | 1804.80 | 1814.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 1717.55 | 1750.03 | 1760.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 1716.74 | 1750.03 | 1760.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-31 11:15:00 | 1627.15 | 1680.68 | 1714.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 1693.30 | 1670.16 | 1669.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1712.60 | 1681.66 | 1675.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 15:15:00 | 1753.00 | 1761.03 | 1745.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 09:15:00 | 1772.15 | 1761.03 | 1745.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1782.10 | 1765.24 | 1749.01 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 1710.60 | 1741.11 | 1744.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 1703.05 | 1733.50 | 1741.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1720.00 | 1696.69 | 1708.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1720.00 | 1696.69 | 1708.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1720.00 | 1696.69 | 1708.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1720.00 | 1696.69 | 1708.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1710.50 | 1699.45 | 1708.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:45:00 | 1731.10 | 1699.45 | 1708.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1707.20 | 1701.96 | 1707.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:45:00 | 1704.30 | 1701.96 | 1707.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1705.60 | 1702.69 | 1707.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:30:00 | 1706.50 | 1702.69 | 1707.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 1709.45 | 1704.04 | 1707.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 15:00:00 | 1709.45 | 1704.04 | 1707.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 1705.00 | 1704.23 | 1707.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:15:00 | 1711.95 | 1704.23 | 1707.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 09:15:00 | 1733.60 | 1710.11 | 1709.99 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 14:15:00 | 1698.45 | 1708.77 | 1710.07 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 1727.85 | 1709.47 | 1709.45 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 13:15:00 | 1705.95 | 1709.46 | 1709.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 14:15:00 | 1687.90 | 1705.15 | 1707.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 15:15:00 | 1664.65 | 1663.80 | 1676.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 09:15:00 | 1702.45 | 1663.80 | 1676.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1679.05 | 1666.85 | 1676.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 10:15:00 | 1671.35 | 1666.85 | 1676.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 11:15:00 | 1672.60 | 1668.68 | 1676.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 12:30:00 | 1672.95 | 1672.60 | 1677.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 13:30:00 | 1675.00 | 1673.07 | 1676.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 1646.10 | 1667.68 | 1674.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:45:00 | 1675.05 | 1667.68 | 1674.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 1653.05 | 1659.84 | 1669.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 10:15:00 | 1645.50 | 1659.84 | 1669.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 13:15:00 | 1675.90 | 1664.40 | 1668.28 | SL hit (close>static) qty=1.00 sl=1673.40 alert=retest2 |

### Cycle 44 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 1676.55 | 1657.47 | 1655.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 1708.00 | 1672.14 | 1663.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 10:15:00 | 1686.80 | 1697.48 | 1682.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 10:15:00 | 1686.80 | 1697.48 | 1682.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1686.80 | 1697.48 | 1682.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:45:00 | 1681.70 | 1697.48 | 1682.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 1684.20 | 1692.54 | 1683.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:00:00 | 1684.20 | 1692.54 | 1683.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 1689.10 | 1691.85 | 1684.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:30:00 | 1687.30 | 1691.85 | 1684.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1694.80 | 1692.84 | 1686.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:45:00 | 1686.20 | 1692.84 | 1686.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1686.50 | 1691.57 | 1686.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 1686.50 | 1691.57 | 1686.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1689.90 | 1691.24 | 1686.98 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 1663.80 | 1684.76 | 1685.27 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 1716.45 | 1685.96 | 1682.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 1759.80 | 1708.43 | 1694.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 14:15:00 | 1785.60 | 1787.22 | 1774.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 14:45:00 | 1787.00 | 1787.22 | 1774.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1785.10 | 1785.73 | 1776.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:30:00 | 1780.90 | 1785.73 | 1776.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 1767.80 | 1782.15 | 1775.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 1767.05 | 1782.15 | 1775.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 1765.65 | 1778.85 | 1774.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:45:00 | 1762.90 | 1778.85 | 1774.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 1784.30 | 1779.66 | 1775.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 14:30:00 | 1791.45 | 1785.61 | 1778.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 10:00:00 | 1790.20 | 1802.74 | 1795.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 13:15:00 | 1767.15 | 1788.38 | 1790.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 1767.15 | 1788.38 | 1790.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 15:15:00 | 1766.10 | 1780.71 | 1786.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 12:15:00 | 1754.55 | 1754.24 | 1763.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:00:00 | 1754.55 | 1754.24 | 1763.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 1692.10 | 1681.64 | 1692.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:00:00 | 1692.10 | 1681.64 | 1692.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 1700.70 | 1685.45 | 1693.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:00:00 | 1700.70 | 1685.45 | 1693.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1715.30 | 1691.42 | 1695.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1715.30 | 1691.42 | 1695.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1715.25 | 1698.71 | 1698.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 1768.90 | 1721.06 | 1711.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 1769.50 | 1772.49 | 1749.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 09:45:00 | 1768.65 | 1772.49 | 1749.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 1833.45 | 1829.47 | 1814.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:45:00 | 1827.40 | 1829.47 | 1814.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1804.75 | 1822.59 | 1814.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 1804.75 | 1822.59 | 1814.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1778.55 | 1813.78 | 1811.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1778.55 | 1813.78 | 1811.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 1785.25 | 1808.07 | 1809.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 1724.55 | 1769.11 | 1782.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 1749.55 | 1744.47 | 1764.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 1749.55 | 1744.47 | 1764.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1766.65 | 1748.90 | 1765.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 1766.65 | 1748.90 | 1765.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1759.95 | 1751.11 | 1764.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 1746.65 | 1751.11 | 1764.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 1659.32 | 1698.12 | 1724.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 1632.10 | 1619.56 | 1656.60 | SL hit (close>ema200) qty=0.50 sl=1619.56 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 1467.75 | 1450.31 | 1450.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 15:15:00 | 1478.00 | 1461.23 | 1455.72 | Break + close above crossover candle high |

### Cycle 51 — SELL (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 09:15:00 | 1306.40 | 1430.26 | 1442.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 13:15:00 | 1280.00 | 1350.26 | 1395.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 1273.00 | 1270.37 | 1312.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 09:30:00 | 1273.50 | 1270.37 | 1312.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1296.80 | 1278.21 | 1305.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1304.25 | 1278.21 | 1305.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1318.20 | 1286.21 | 1307.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:00:00 | 1318.20 | 1286.21 | 1307.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 1326.25 | 1294.22 | 1308.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:45:00 | 1328.05 | 1294.22 | 1308.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 10:15:00 | 1391.50 | 1327.66 | 1321.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 11:15:00 | 1402.80 | 1342.69 | 1328.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 12:15:00 | 1380.70 | 1384.96 | 1363.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 12:30:00 | 1375.75 | 1384.96 | 1363.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1385.60 | 1397.96 | 1388.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 1387.70 | 1397.96 | 1388.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1377.90 | 1393.94 | 1387.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 1377.90 | 1393.94 | 1387.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1369.05 | 1388.97 | 1385.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 1369.05 | 1388.97 | 1385.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 1349.40 | 1377.19 | 1380.54 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 15:15:00 | 1385.60 | 1376.28 | 1375.93 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1370.00 | 1375.02 | 1375.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 1359.30 | 1370.66 | 1373.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1307.45 | 1304.64 | 1327.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 10:45:00 | 1303.85 | 1304.64 | 1327.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1242.40 | 1226.93 | 1242.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1242.40 | 1226.93 | 1242.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1243.85 | 1230.31 | 1242.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 1229.80 | 1230.31 | 1242.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1220.95 | 1228.44 | 1240.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 1216.35 | 1228.44 | 1240.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 1268.15 | 1230.44 | 1233.76 | SL hit (close>static) qty=1.00 sl=1245.95 alert=retest2 |

### Cycle 56 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 1270.40 | 1238.43 | 1237.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 1282.95 | 1259.14 | 1247.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1273.55 | 1275.09 | 1266.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:15:00 | 1272.40 | 1275.09 | 1266.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1269.35 | 1273.94 | 1266.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 1269.35 | 1273.94 | 1266.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 1269.00 | 1272.95 | 1266.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:45:00 | 1281.90 | 1274.66 | 1268.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 15:15:00 | 1260.00 | 1270.51 | 1267.66 | SL hit (close<static) qty=1.00 sl=1266.50 alert=retest2 |

### Cycle 57 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 1397.15 | 1416.19 | 1418.07 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 1428.75 | 1420.77 | 1419.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 11:15:00 | 1442.30 | 1425.07 | 1421.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1479.35 | 1489.76 | 1476.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 1479.35 | 1489.76 | 1476.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1479.35 | 1489.76 | 1476.02 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2025-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 09:15:00 | 1430.90 | 1467.55 | 1470.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 13:15:00 | 1416.00 | 1442.85 | 1456.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 1444.35 | 1439.84 | 1451.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-24 09:45:00 | 1447.30 | 1439.84 | 1451.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 1445.55 | 1440.99 | 1450.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 10:30:00 | 1449.50 | 1440.99 | 1450.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 11:15:00 | 1453.15 | 1443.42 | 1451.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 12:00:00 | 1453.15 | 1443.42 | 1451.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 1445.30 | 1443.79 | 1450.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 14:45:00 | 1437.90 | 1445.49 | 1450.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 1440.30 | 1446.60 | 1450.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:30:00 | 1437.35 | 1443.52 | 1448.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 09:30:00 | 1439.30 | 1427.16 | 1429.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 1435.10 | 1428.75 | 1429.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:45:00 | 1441.05 | 1428.75 | 1429.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-28 11:15:00 | 1437.50 | 1430.50 | 1430.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 1437.50 | 1430.50 | 1430.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 1461.85 | 1438.73 | 1434.42 | Break + close above crossover candle high |

### Cycle 61 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 1383.70 | 1431.29 | 1432.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 12:15:00 | 1363.75 | 1400.84 | 1416.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 13:15:00 | 1350.70 | 1347.04 | 1362.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 14:00:00 | 1350.70 | 1347.04 | 1362.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1301.50 | 1293.35 | 1308.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 1298.55 | 1297.27 | 1308.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1297.50 | 1311.62 | 1312.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 09:30:00 | 1297.40 | 1288.13 | 1292.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 09:15:00 | 1311.50 | 1288.80 | 1286.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 1311.50 | 1288.80 | 1286.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 14:15:00 | 1319.00 | 1306.32 | 1296.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 11:15:00 | 1307.60 | 1310.03 | 1302.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 11:30:00 | 1312.30 | 1310.03 | 1302.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 12:15:00 | 1321.50 | 1312.32 | 1303.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 13:15:00 | 1345.90 | 1312.32 | 1303.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 1302.00 | 1324.00 | 1313.56 | SL hit (close<static) qty=1.00 sl=1302.50 alert=retest2 |

### Cycle 63 — SELL (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 13:15:00 | 1297.00 | 1307.21 | 1307.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 10:15:00 | 1287.90 | 1299.55 | 1303.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 13:15:00 | 1246.10 | 1242.97 | 1256.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 13:45:00 | 1244.20 | 1242.97 | 1256.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1251.20 | 1245.52 | 1254.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:30:00 | 1246.80 | 1246.78 | 1254.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 11:15:00 | 1268.10 | 1251.04 | 1255.38 | SL hit (close>static) qty=1.00 sl=1261.30 alert=retest2 |

### Cycle 64 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 1264.50 | 1257.92 | 1257.82 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 1237.60 | 1254.70 | 1256.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 1210.60 | 1230.54 | 1240.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1221.50 | 1217.84 | 1229.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1221.50 | 1217.84 | 1229.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1221.50 | 1217.84 | 1229.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1221.50 | 1217.84 | 1229.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1233.30 | 1221.76 | 1229.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1233.30 | 1221.76 | 1229.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1240.70 | 1225.55 | 1230.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:00:00 | 1240.70 | 1225.55 | 1230.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 1239.50 | 1233.48 | 1233.17 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 1225.50 | 1231.89 | 1232.47 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 1237.00 | 1233.09 | 1232.93 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 1227.80 | 1232.40 | 1232.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 1212.50 | 1228.42 | 1230.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 1232.20 | 1228.73 | 1230.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 1232.20 | 1228.73 | 1230.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1232.20 | 1228.73 | 1230.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 1232.20 | 1228.73 | 1230.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1240.50 | 1231.08 | 1231.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:30:00 | 1241.70 | 1231.08 | 1231.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 1251.30 | 1235.12 | 1233.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 10:15:00 | 1261.50 | 1244.72 | 1238.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 12:15:00 | 1241.40 | 1246.31 | 1240.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 12:15:00 | 1241.40 | 1246.31 | 1240.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 1241.40 | 1246.31 | 1240.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 1241.40 | 1246.31 | 1240.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1225.80 | 1242.21 | 1239.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 1225.80 | 1242.21 | 1239.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1227.60 | 1239.29 | 1238.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 1221.40 | 1239.29 | 1238.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1212.10 | 1233.85 | 1235.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 1204.10 | 1227.90 | 1232.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 1232.20 | 1225.16 | 1230.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 11:15:00 | 1232.20 | 1225.16 | 1230.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 1232.20 | 1225.16 | 1230.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:00:00 | 1232.20 | 1225.16 | 1230.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 1235.60 | 1227.25 | 1231.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 13:15:00 | 1231.60 | 1227.25 | 1231.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 1235.90 | 1228.98 | 1231.52 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1259.70 | 1236.92 | 1234.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1266.70 | 1246.44 | 1239.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1262.60 | 1262.77 | 1254.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:30:00 | 1263.00 | 1262.77 | 1254.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1258.00 | 1260.48 | 1255.45 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 14:15:00 | 1248.00 | 1253.07 | 1253.44 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 1259.30 | 1254.15 | 1253.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 1272.80 | 1262.11 | 1258.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1258.80 | 1267.66 | 1264.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1258.80 | 1267.66 | 1264.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1258.80 | 1267.66 | 1264.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 1257.00 | 1267.66 | 1264.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1266.90 | 1267.50 | 1264.77 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1239.80 | 1259.24 | 1261.62 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 1266.80 | 1262.15 | 1261.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 1268.00 | 1263.32 | 1262.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 1265.80 | 1265.81 | 1264.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 11:00:00 | 1265.80 | 1265.81 | 1264.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1271.90 | 1269.19 | 1266.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 1268.20 | 1269.19 | 1266.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1267.50 | 1269.41 | 1267.12 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 1262.00 | 1266.52 | 1266.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 1254.00 | 1264.01 | 1265.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 1260.00 | 1257.39 | 1260.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1260.00 | 1257.39 | 1260.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1260.00 | 1257.39 | 1260.63 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 1267.30 | 1261.12 | 1260.50 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 1253.60 | 1261.51 | 1261.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 13:15:00 | 1246.20 | 1255.64 | 1258.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 10:15:00 | 1235.20 | 1231.45 | 1237.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 10:15:00 | 1235.20 | 1231.45 | 1237.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1235.20 | 1231.45 | 1237.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:30:00 | 1236.60 | 1231.45 | 1237.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1236.50 | 1232.46 | 1236.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 1236.50 | 1232.46 | 1236.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 1245.80 | 1235.13 | 1237.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:00:00 | 1245.80 | 1235.13 | 1237.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1242.80 | 1236.66 | 1238.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:30:00 | 1241.40 | 1236.66 | 1238.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 15:15:00 | 1251.00 | 1240.64 | 1239.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 1257.50 | 1245.80 | 1242.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 1305.50 | 1306.53 | 1291.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:45:00 | 1303.00 | 1306.53 | 1291.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1302.60 | 1309.50 | 1302.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:30:00 | 1302.30 | 1309.50 | 1302.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1303.70 | 1308.34 | 1302.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1303.70 | 1308.34 | 1302.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1302.40 | 1307.15 | 1302.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1302.40 | 1307.15 | 1302.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1283.80 | 1302.48 | 1300.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1283.80 | 1302.48 | 1300.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1291.50 | 1300.29 | 1299.70 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 1287.90 | 1297.81 | 1298.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 10:15:00 | 1280.90 | 1292.73 | 1296.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1286.00 | 1283.12 | 1288.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 1286.00 | 1283.12 | 1288.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1288.90 | 1284.28 | 1288.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 1288.90 | 1284.28 | 1288.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1307.60 | 1288.94 | 1290.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1307.60 | 1288.94 | 1290.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 1309.40 | 1293.03 | 1291.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 1313.90 | 1297.21 | 1293.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 1290.00 | 1306.92 | 1302.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 14:15:00 | 1290.00 | 1306.92 | 1302.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 1290.00 | 1306.92 | 1302.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 1290.00 | 1306.92 | 1302.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 1290.00 | 1303.53 | 1301.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 1283.10 | 1303.53 | 1301.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 1279.80 | 1298.79 | 1299.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 1260.00 | 1272.15 | 1281.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 1298.50 | 1273.26 | 1277.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 1298.50 | 1273.26 | 1277.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1298.50 | 1273.26 | 1277.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1298.50 | 1273.26 | 1277.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1290.00 | 1276.61 | 1278.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1274.40 | 1276.61 | 1278.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1288.00 | 1279.49 | 1279.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 1288.00 | 1279.49 | 1279.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 1294.30 | 1284.65 | 1281.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1306.30 | 1308.45 | 1298.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1306.30 | 1308.45 | 1298.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1303.30 | 1306.25 | 1300.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 14:00:00 | 1310.00 | 1305.86 | 1301.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:15:00 | 1310.60 | 1312.08 | 1310.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:30:00 | 1315.90 | 1311.45 | 1310.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 1302.60 | 1309.68 | 1309.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 1302.60 | 1309.68 | 1309.94 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 1320.00 | 1311.29 | 1310.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 09:15:00 | 1322.30 | 1314.52 | 1312.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 1359.50 | 1366.83 | 1353.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 1365.30 | 1366.83 | 1353.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1357.60 | 1364.20 | 1354.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:30:00 | 1364.70 | 1364.52 | 1355.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 1350.90 | 1361.71 | 1355.91 | SL hit (close<static) qty=1.00 sl=1353.30 alert=retest2 |

### Cycle 87 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1360.70 | 1365.67 | 1366.25 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 12:15:00 | 1370.10 | 1367.25 | 1366.91 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 1363.30 | 1367.22 | 1367.39 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 1370.60 | 1367.89 | 1367.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 1375.40 | 1369.39 | 1368.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1388.70 | 1389.11 | 1381.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1388.70 | 1389.11 | 1381.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1388.70 | 1389.11 | 1381.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 1385.80 | 1389.11 | 1381.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1390.10 | 1389.45 | 1383.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 1391.20 | 1389.98 | 1383.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:00:00 | 1392.10 | 1389.98 | 1383.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:45:00 | 1391.10 | 1391.87 | 1386.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:15:00 | 1391.40 | 1391.13 | 1387.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 1390.70 | 1391.42 | 1388.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 1391.90 | 1391.42 | 1388.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1390.00 | 1391.13 | 1388.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:15:00 | 1382.10 | 1391.13 | 1388.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1381.00 | 1389.11 | 1388.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 1381.00 | 1389.11 | 1388.21 | SL hit (close<static) qty=1.00 sl=1382.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1377.20 | 1385.68 | 1386.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 1370.50 | 1381.45 | 1384.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 1386.50 | 1382.24 | 1384.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 1386.50 | 1382.24 | 1384.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1386.50 | 1382.24 | 1384.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 1386.50 | 1382.24 | 1384.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1387.70 | 1383.33 | 1384.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1387.70 | 1383.33 | 1384.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1388.90 | 1384.45 | 1385.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 1391.50 | 1384.45 | 1385.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1386.30 | 1384.82 | 1385.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1357.00 | 1385.05 | 1385.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 1337.10 | 1329.63 | 1329.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1337.10 | 1329.63 | 1329.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 11:15:00 | 1343.40 | 1333.76 | 1331.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 1324.10 | 1335.16 | 1332.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 1324.10 | 1335.16 | 1332.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1324.10 | 1335.16 | 1332.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 1324.10 | 1335.16 | 1332.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1320.20 | 1332.16 | 1331.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1320.30 | 1332.16 | 1331.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1314.60 | 1328.65 | 1330.22 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1340.60 | 1328.72 | 1327.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 1342.90 | 1331.56 | 1328.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 1331.00 | 1331.44 | 1328.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 1331.00 | 1331.44 | 1328.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1331.00 | 1331.44 | 1328.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:45:00 | 1330.40 | 1331.44 | 1328.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1333.60 | 1331.88 | 1329.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 1333.50 | 1331.88 | 1329.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1327.30 | 1330.96 | 1329.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 1327.30 | 1330.96 | 1329.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1324.90 | 1329.75 | 1328.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 1324.30 | 1329.75 | 1328.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 1316.40 | 1327.08 | 1327.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 15:15:00 | 1314.00 | 1323.17 | 1325.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 15:15:00 | 1317.00 | 1314.67 | 1318.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 15:15:00 | 1317.00 | 1314.67 | 1318.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1317.00 | 1314.67 | 1318.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1312.30 | 1316.73 | 1319.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1319.40 | 1317.27 | 1319.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:30:00 | 1306.30 | 1314.79 | 1318.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1240.98 | 1293.94 | 1303.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 1257.90 | 1250.11 | 1265.94 | SL hit (close>ema200) qty=0.50 sl=1250.11 alert=retest2 |

### Cycle 96 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 1273.70 | 1266.63 | 1266.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 1356.40 | 1288.85 | 1277.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 1367.00 | 1367.74 | 1348.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 1367.00 | 1367.74 | 1348.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1352.50 | 1364.77 | 1354.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 1353.60 | 1364.77 | 1354.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1360.00 | 1363.82 | 1355.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:00:00 | 1366.00 | 1362.46 | 1356.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 1365.40 | 1362.41 | 1356.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:45:00 | 1364.90 | 1363.87 | 1359.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 1366.10 | 1360.07 | 1358.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1362.00 | 1362.94 | 1360.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 1362.60 | 1362.94 | 1360.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1355.90 | 1361.53 | 1360.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 1355.90 | 1361.53 | 1360.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1358.20 | 1360.86 | 1360.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1350.30 | 1360.86 | 1360.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1348.00 | 1358.29 | 1359.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1348.00 | 1358.29 | 1359.05 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 14:15:00 | 1361.60 | 1357.01 | 1356.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 10:15:00 | 1373.30 | 1360.49 | 1358.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1417.80 | 1426.36 | 1411.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1417.80 | 1426.36 | 1411.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1421.40 | 1424.60 | 1416.77 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1399.10 | 1413.34 | 1414.26 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 1422.50 | 1415.17 | 1415.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 12:15:00 | 1434.90 | 1421.55 | 1418.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 10:15:00 | 1419.80 | 1426.06 | 1422.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 10:15:00 | 1419.80 | 1426.06 | 1422.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1419.80 | 1426.06 | 1422.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 1416.80 | 1426.06 | 1422.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 1424.10 | 1425.67 | 1422.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:45:00 | 1420.30 | 1425.67 | 1422.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1423.40 | 1425.21 | 1422.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:45:00 | 1422.40 | 1425.21 | 1422.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 1416.50 | 1423.47 | 1421.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 1416.50 | 1423.47 | 1421.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1419.60 | 1422.70 | 1421.75 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 1410.50 | 1419.15 | 1420.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 1408.70 | 1413.58 | 1416.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 13:15:00 | 1394.70 | 1391.83 | 1399.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 1394.70 | 1391.83 | 1399.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1402.80 | 1395.17 | 1399.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 1399.70 | 1395.17 | 1399.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1400.70 | 1396.28 | 1399.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 15:15:00 | 1398.00 | 1400.31 | 1400.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1409.60 | 1401.80 | 1401.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1409.60 | 1401.80 | 1401.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 1415.80 | 1407.38 | 1404.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 13:15:00 | 1417.40 | 1418.74 | 1413.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 1417.40 | 1418.74 | 1413.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1413.80 | 1417.75 | 1413.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:45:00 | 1414.00 | 1417.75 | 1413.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1417.00 | 1417.60 | 1413.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1414.20 | 1417.60 | 1413.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1408.80 | 1415.84 | 1413.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 1409.90 | 1415.84 | 1413.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1408.40 | 1414.35 | 1412.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:45:00 | 1407.10 | 1414.35 | 1412.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 1406.10 | 1410.94 | 1411.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 15:15:00 | 1404.20 | 1409.41 | 1410.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 1410.50 | 1408.81 | 1410.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 10:15:00 | 1410.50 | 1408.81 | 1410.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1410.50 | 1408.81 | 1410.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 1412.00 | 1408.81 | 1410.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 1426.80 | 1412.41 | 1411.56 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1393.30 | 1415.06 | 1415.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 1386.30 | 1409.31 | 1413.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 1375.40 | 1375.21 | 1387.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 1375.40 | 1375.21 | 1387.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1370.50 | 1370.35 | 1376.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1357.60 | 1372.43 | 1375.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 1360.00 | 1353.54 | 1359.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 1359.20 | 1355.31 | 1360.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 13:15:00 | 1360.20 | 1355.31 | 1360.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1373.80 | 1359.01 | 1361.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 1373.80 | 1359.01 | 1361.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1374.50 | 1362.11 | 1362.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 1375.90 | 1362.11 | 1362.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-29 15:15:00 | 1381.90 | 1366.06 | 1364.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 15:15:00 | 1381.90 | 1366.06 | 1364.36 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 1359.20 | 1363.42 | 1363.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 14:15:00 | 1352.00 | 1361.13 | 1362.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 15:15:00 | 1354.30 | 1353.97 | 1357.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 09:15:00 | 1352.60 | 1353.97 | 1357.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1351.80 | 1353.53 | 1356.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 1355.60 | 1353.53 | 1356.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 1350.50 | 1352.86 | 1355.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 1350.50 | 1352.86 | 1355.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 1355.00 | 1353.29 | 1355.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:45:00 | 1355.40 | 1353.29 | 1355.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 1355.10 | 1353.65 | 1355.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:45:00 | 1355.30 | 1353.65 | 1355.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 1358.40 | 1354.60 | 1355.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:45:00 | 1362.90 | 1354.60 | 1355.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 1358.00 | 1355.28 | 1356.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 1354.90 | 1355.28 | 1356.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1351.80 | 1353.90 | 1355.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 1348.50 | 1353.90 | 1355.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 13:15:00 | 1357.70 | 1352.33 | 1353.88 | SL hit (close>static) qty=1.00 sl=1356.80 alert=retest2 |

### Cycle 108 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 1366.20 | 1356.86 | 1355.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 09:15:00 | 1386.50 | 1370.88 | 1366.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 15:15:00 | 1377.30 | 1378.47 | 1373.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:15:00 | 1412.00 | 1378.47 | 1373.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1391.00 | 1408.18 | 1396.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 1391.00 | 1408.18 | 1396.22 | SL hit (close<ema400) qty=1.00 sl=1396.22 alert=retest1 |

### Cycle 109 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 1371.30 | 1387.32 | 1388.80 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 15:15:00 | 1395.10 | 1387.91 | 1387.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 1396.90 | 1391.58 | 1389.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 1410.80 | 1421.85 | 1415.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 1410.80 | 1421.85 | 1415.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1410.80 | 1421.85 | 1415.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:00:00 | 1423.30 | 1422.14 | 1416.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 1427.10 | 1438.52 | 1436.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 1432.80 | 1435.92 | 1435.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 11:15:00 | 1432.80 | 1435.92 | 1435.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 1415.70 | 1430.32 | 1433.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1427.50 | 1421.72 | 1426.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 1427.50 | 1421.72 | 1426.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1427.50 | 1421.72 | 1426.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 1427.50 | 1421.72 | 1426.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1426.00 | 1422.57 | 1426.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1410.90 | 1422.57 | 1426.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 1340.36 | 1348.44 | 1364.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 1316.00 | 1315.25 | 1336.00 | SL hit (close>ema200) qty=0.50 sl=1315.25 alert=retest2 |

### Cycle 112 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1348.20 | 1337.68 | 1337.26 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1320.20 | 1336.28 | 1337.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 11:15:00 | 1310.90 | 1327.48 | 1332.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1321.60 | 1316.26 | 1324.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 1321.60 | 1316.26 | 1324.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1330.90 | 1319.19 | 1324.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 1330.90 | 1319.19 | 1324.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 1322.90 | 1319.93 | 1324.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:45:00 | 1334.40 | 1319.93 | 1324.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1323.40 | 1320.62 | 1324.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:30:00 | 1320.40 | 1320.62 | 1324.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 1335.40 | 1323.58 | 1325.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 1335.40 | 1323.58 | 1325.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 1339.90 | 1326.84 | 1326.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 09:15:00 | 1353.30 | 1335.74 | 1331.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 14:15:00 | 1349.40 | 1350.19 | 1341.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 1349.40 | 1350.19 | 1341.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1397.80 | 1405.87 | 1399.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 1397.10 | 1405.87 | 1399.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 1393.10 | 1403.32 | 1398.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 1388.40 | 1403.32 | 1398.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 1392.20 | 1401.09 | 1398.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:45:00 | 1386.40 | 1401.09 | 1398.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 1382.60 | 1394.50 | 1395.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 1366.50 | 1388.90 | 1393.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 15:15:00 | 1365.50 | 1360.59 | 1369.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:15:00 | 1367.80 | 1360.59 | 1369.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1374.50 | 1363.38 | 1369.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1374.50 | 1363.38 | 1369.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1379.40 | 1366.58 | 1370.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 1379.40 | 1366.58 | 1370.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1382.90 | 1369.84 | 1371.91 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1387.70 | 1373.42 | 1373.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1390.20 | 1376.77 | 1374.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 1368.90 | 1377.58 | 1375.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 1368.90 | 1377.58 | 1375.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1368.90 | 1377.58 | 1375.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 1386.70 | 1379.54 | 1377.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 1365.70 | 1376.67 | 1377.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 1365.70 | 1376.67 | 1377.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 1360.40 | 1369.29 | 1372.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 1327.50 | 1326.60 | 1333.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:15:00 | 1318.10 | 1324.60 | 1331.08 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:45:00 | 1318.10 | 1323.16 | 1329.84 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 15:15:00 | 1318.00 | 1320.41 | 1326.71 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1321.80 | 1320.43 | 1325.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1336.10 | 1324.18 | 1326.04 | SL hit (close>ema400) qty=1.00 sl=1326.04 alert=retest1 |

### Cycle 118 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1335.20 | 1328.82 | 1327.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 1339.90 | 1332.26 | 1329.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 1330.30 | 1335.65 | 1332.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 15:15:00 | 1330.30 | 1335.65 | 1332.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1330.30 | 1335.65 | 1332.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1338.70 | 1335.65 | 1332.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1344.40 | 1337.40 | 1333.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 1346.80 | 1337.40 | 1333.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:00:00 | 1351.50 | 1341.39 | 1336.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 1368.90 | 1387.47 | 1389.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 13:15:00 | 1368.90 | 1387.47 | 1389.04 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 1406.80 | 1389.94 | 1387.68 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 1384.50 | 1391.02 | 1391.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1364.30 | 1382.04 | 1386.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 1364.00 | 1360.93 | 1369.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:00:00 | 1364.00 | 1360.93 | 1369.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1366.80 | 1362.11 | 1369.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 1369.40 | 1362.11 | 1369.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1361.60 | 1362.00 | 1368.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 1361.60 | 1362.00 | 1368.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1377.30 | 1365.03 | 1368.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 1382.90 | 1365.03 | 1368.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1376.30 | 1367.29 | 1369.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 1377.10 | 1367.29 | 1369.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 1381.20 | 1372.30 | 1371.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 1385.00 | 1376.57 | 1373.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1475.30 | 1482.17 | 1458.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 1475.30 | 1482.17 | 1458.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1479.70 | 1496.77 | 1482.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1479.70 | 1496.77 | 1482.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1473.00 | 1492.02 | 1481.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1473.00 | 1492.02 | 1481.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1478.60 | 1489.33 | 1481.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:45:00 | 1473.70 | 1489.33 | 1481.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1474.30 | 1484.26 | 1480.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 1473.60 | 1484.26 | 1480.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 1466.00 | 1478.40 | 1478.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 1460.80 | 1472.32 | 1475.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1475.30 | 1471.02 | 1473.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 11:15:00 | 1475.30 | 1471.02 | 1473.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 1475.30 | 1471.02 | 1473.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:00:00 | 1475.30 | 1471.02 | 1473.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1470.00 | 1470.82 | 1473.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 1473.00 | 1470.82 | 1473.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1451.80 | 1465.81 | 1469.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 1446.40 | 1459.66 | 1466.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 1447.00 | 1455.70 | 1463.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1445.50 | 1452.53 | 1460.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1374.65 | 1397.75 | 1413.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 1374.08 | 1389.32 | 1408.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 1373.22 | 1389.32 | 1408.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 09:15:00 | 1301.76 | 1342.48 | 1375.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 124 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 1340.80 | 1317.96 | 1317.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 1345.30 | 1330.22 | 1323.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 1340.00 | 1349.96 | 1338.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 1340.00 | 1349.96 | 1338.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1340.00 | 1349.96 | 1338.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 1336.50 | 1349.96 | 1338.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 1349.90 | 1351.55 | 1343.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 1310.20 | 1351.55 | 1343.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1351.90 | 1351.62 | 1344.64 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 1314.70 | 1339.31 | 1340.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 1301.50 | 1320.11 | 1328.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 1312.60 | 1311.17 | 1321.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:45:00 | 1309.00 | 1311.17 | 1321.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1330.00 | 1308.25 | 1315.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1330.00 | 1308.25 | 1315.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1327.00 | 1312.00 | 1316.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1356.50 | 1312.00 | 1316.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1360.90 | 1321.78 | 1320.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1406.60 | 1364.65 | 1345.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1409.10 | 1418.62 | 1400.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:30:00 | 1408.60 | 1418.62 | 1400.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1467.70 | 1483.58 | 1462.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:00:00 | 1467.70 | 1483.58 | 1462.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 1522.60 | 1525.55 | 1517.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:45:00 | 1521.60 | 1525.55 | 1517.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1521.00 | 1526.90 | 1521.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:45:00 | 1520.00 | 1526.90 | 1521.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1517.40 | 1525.00 | 1521.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 1517.40 | 1525.00 | 1521.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1524.30 | 1524.86 | 1521.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 15:00:00 | 1527.90 | 1525.28 | 1522.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 1528.00 | 1526.26 | 1523.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:45:00 | 1527.60 | 1528.18 | 1524.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 1510.90 | 1529.75 | 1529.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 1510.90 | 1529.75 | 1529.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1504.00 | 1524.60 | 1527.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 1535.00 | 1526.58 | 1527.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 1535.00 | 1526.58 | 1527.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 1535.00 | 1526.58 | 1527.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 1536.00 | 1526.58 | 1527.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 1543.90 | 1530.05 | 1529.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 14:15:00 | 1546.70 | 1535.91 | 1532.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 10:15:00 | 1539.40 | 1540.46 | 1535.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 11:00:00 | 1539.40 | 1540.46 | 1535.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1535.30 | 1539.43 | 1535.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 1533.90 | 1539.43 | 1535.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 1536.50 | 1538.84 | 1535.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:30:00 | 1535.70 | 1538.84 | 1535.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 1539.00 | 1538.88 | 1535.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:15:00 | 1535.60 | 1538.88 | 1535.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1535.70 | 1538.24 | 1535.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:30:00 | 1531.40 | 1538.24 | 1535.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1537.00 | 1537.99 | 1536.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1545.90 | 1539.67 | 1536.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 1535.00 | 1541.53 | 1539.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 1532.70 | 1541.53 | 1539.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 1534.90 | 1540.21 | 1538.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 1540.20 | 1542.57 | 1540.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 1522.90 | 1537.07 | 1538.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 1522.90 | 1537.07 | 1538.06 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1549.60 | 1536.99 | 1536.91 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 1535.40 | 1536.67 | 1536.77 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 12:15:00 | 1537.50 | 1536.84 | 1536.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 1547.40 | 1539.36 | 1538.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 15:15:00 | 1557.50 | 1557.76 | 1550.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 09:15:00 | 1505.70 | 1557.76 | 1550.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1522.90 | 1550.79 | 1547.67 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 1509.10 | 1538.65 | 1542.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1496.60 | 1530.24 | 1538.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 1455.70 | 1454.51 | 1480.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 1455.70 | 1454.51 | 1480.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 1478.90 | 1460.76 | 1478.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 1478.90 | 1460.76 | 1478.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1476.30 | 1463.87 | 1478.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 1479.20 | 1463.87 | 1478.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1480.80 | 1467.25 | 1478.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1480.80 | 1467.25 | 1478.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1484.70 | 1470.74 | 1479.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1485.90 | 1470.74 | 1479.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1482.30 | 1473.05 | 1479.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 1485.00 | 1473.05 | 1479.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1469.50 | 1472.34 | 1478.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:30:00 | 1464.00 | 1470.43 | 1477.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1420.30 | 1476.07 | 1478.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 1490.00 | 1462.53 | 1459.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 1490.00 | 1462.53 | 1459.96 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1426.50 | 1460.21 | 1461.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1396.80 | 1442.25 | 1451.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1408.00 | 1398.19 | 1413.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1408.00 | 1398.19 | 1413.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1408.00 | 1398.19 | 1413.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1415.40 | 1398.19 | 1413.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1413.40 | 1402.48 | 1413.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 1415.00 | 1402.48 | 1413.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1404.50 | 1402.89 | 1412.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1398.40 | 1402.89 | 1412.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 1428.20 | 1416.51 | 1415.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1428.20 | 1416.51 | 1415.10 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1372.70 | 1411.11 | 1413.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 1361.00 | 1401.09 | 1409.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 1265.20 | 1263.24 | 1297.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 10:45:00 | 1267.70 | 1263.24 | 1297.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1301.50 | 1272.95 | 1296.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1300.00 | 1272.95 | 1296.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1304.00 | 1279.16 | 1297.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1303.80 | 1279.16 | 1297.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1360.60 | 1313.00 | 1308.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1370.30 | 1337.31 | 1321.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1331.90 | 1346.78 | 1330.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1331.90 | 1346.78 | 1330.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1331.90 | 1346.78 | 1330.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1331.90 | 1346.78 | 1330.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1332.20 | 1343.86 | 1331.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 1326.00 | 1343.86 | 1331.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1326.20 | 1340.33 | 1330.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 1326.20 | 1340.33 | 1330.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1332.40 | 1338.75 | 1330.73 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1296.20 | 1324.79 | 1326.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1277.70 | 1315.37 | 1321.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1291.30 | 1290.13 | 1303.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1291.30 | 1290.13 | 1303.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1291.30 | 1290.13 | 1303.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1283.20 | 1290.13 | 1303.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 1263.90 | 1282.78 | 1299.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 14:15:00 | 1219.04 | 1260.34 | 1282.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1200.70 | 1244.48 | 1271.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1231.30 | 1229.49 | 1254.01 | SL hit (close>ema200) qty=0.50 sl=1229.49 alert=retest2 |

### Cycle 140 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 1271.30 | 1239.80 | 1239.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 14:15:00 | 1282.20 | 1263.47 | 1255.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 1389.90 | 1392.73 | 1369.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 13:00:00 | 1389.90 | 1392.73 | 1369.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1435.40 | 1469.53 | 1460.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 1435.40 | 1469.53 | 1460.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1448.50 | 1465.33 | 1459.01 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 1447.10 | 1454.50 | 1455.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1436.00 | 1447.55 | 1451.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 1459.00 | 1448.44 | 1450.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 13:15:00 | 1459.00 | 1448.44 | 1450.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1459.00 | 1448.44 | 1450.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 1459.00 | 1448.44 | 1450.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1465.70 | 1451.90 | 1452.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 1465.70 | 1451.90 | 1452.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 1455.00 | 1452.52 | 1452.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 1493.90 | 1460.79 | 1456.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 1503.80 | 1505.87 | 1492.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 13:45:00 | 1506.00 | 1505.87 | 1492.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 1489.60 | 1505.38 | 1497.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 1489.60 | 1505.38 | 1497.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1497.30 | 1503.77 | 1497.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:30:00 | 1489.60 | 1503.77 | 1497.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1484.50 | 1499.91 | 1496.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1484.50 | 1499.91 | 1496.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1478.20 | 1495.57 | 1494.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1478.20 | 1495.57 | 1494.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 1471.00 | 1490.66 | 1492.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 1423.90 | 1477.31 | 1486.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 1450.00 | 1442.77 | 1458.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 11:00:00 | 1450.00 | 1442.77 | 1458.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1446.00 | 1443.41 | 1457.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 1454.30 | 1443.41 | 1457.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 1454.10 | 1448.54 | 1456.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 1424.80 | 1449.31 | 1455.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:15:00 | 1353.56 | 1374.37 | 1394.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 1373.10 | 1371.45 | 1389.41 | SL hit (close>ema200) qty=0.50 sl=1371.45 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 09:30:00 | 1282.95 | 2024-05-13 15:15:00 | 1301.85 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-05-28 13:15:00 | 1400.05 | 2024-05-29 14:15:00 | 1376.30 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-05-28 14:30:00 | 1398.40 | 2024-05-29 14:15:00 | 1376.30 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-05-29 10:30:00 | 1397.45 | 2024-05-29 14:15:00 | 1376.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-05-31 12:30:00 | 1371.40 | 2024-06-03 09:15:00 | 1423.25 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2024-05-31 14:15:00 | 1365.60 | 2024-06-03 09:15:00 | 1423.25 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2024-06-10 11:15:00 | 1462.00 | 2024-06-11 13:15:00 | 1446.05 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-06-10 12:00:00 | 1459.55 | 2024-06-12 14:15:00 | 1447.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-06-10 13:15:00 | 1458.90 | 2024-06-12 14:15:00 | 1447.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-06-10 14:45:00 | 1461.80 | 2024-06-12 14:15:00 | 1447.10 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-06-11 10:30:00 | 1471.10 | 2024-06-12 14:15:00 | 1447.10 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-06-27 12:30:00 | 1494.05 | 2024-07-02 12:15:00 | 1419.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-27 13:30:00 | 1493.80 | 2024-07-02 12:15:00 | 1419.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-28 09:30:00 | 1492.50 | 2024-07-02 12:15:00 | 1417.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-27 12:30:00 | 1494.05 | 2024-07-02 14:15:00 | 1450.65 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2024-06-27 13:30:00 | 1493.80 | 2024-07-02 14:15:00 | 1450.65 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2024-06-28 09:30:00 | 1492.50 | 2024-07-02 14:15:00 | 1450.65 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2024-07-22 12:45:00 | 1482.75 | 2024-07-24 10:15:00 | 1511.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-07-22 13:30:00 | 1478.65 | 2024-07-24 10:15:00 | 1511.00 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-07-23 09:15:00 | 1481.95 | 2024-07-24 10:15:00 | 1511.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-07-23 15:00:00 | 1481.40 | 2024-07-24 10:15:00 | 1511.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-08-06 14:00:00 | 1469.10 | 2024-08-08 11:15:00 | 1494.40 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-08-07 10:00:00 | 1469.95 | 2024-08-08 11:15:00 | 1494.40 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-08-07 13:15:00 | 1466.65 | 2024-08-08 11:15:00 | 1494.40 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-08-08 09:15:00 | 1456.15 | 2024-08-08 11:15:00 | 1494.40 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-08-08 13:15:00 | 1435.00 | 2024-08-12 09:15:00 | 1528.20 | STOP_HIT | 1.00 | -6.49% |
| SELL | retest2 | 2024-08-08 14:45:00 | 1444.35 | 2024-08-12 09:15:00 | 1528.20 | STOP_HIT | 1.00 | -5.81% |
| BUY | retest2 | 2024-08-16 09:15:00 | 1581.80 | 2024-08-16 12:15:00 | 1531.90 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2024-08-16 14:00:00 | 1554.05 | 2024-08-16 14:15:00 | 1535.90 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-09-02 09:15:00 | 1764.60 | 2024-09-05 15:15:00 | 1781.90 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2024-09-02 11:15:00 | 1768.60 | 2024-09-05 15:15:00 | 1781.90 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2024-09-02 15:15:00 | 1790.80 | 2024-09-05 15:15:00 | 1781.90 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-09-09 09:15:00 | 1770.70 | 2024-09-09 09:15:00 | 1791.90 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-09-20 11:15:00 | 1932.10 | 2024-09-23 13:15:00 | 1887.55 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-09-27 14:00:00 | 1854.40 | 2024-10-01 15:15:00 | 1860.65 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-09-30 11:00:00 | 1854.80 | 2024-10-07 10:15:00 | 1761.68 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2024-09-30 11:30:00 | 1854.70 | 2024-10-07 10:15:00 | 1762.06 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2024-10-01 10:00:00 | 1854.85 | 2024-10-07 10:15:00 | 1761.96 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2024-10-01 14:30:00 | 1847.85 | 2024-10-07 10:15:00 | 1762.11 | PARTIAL | 0.50 | 4.64% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1831.85 | 2024-10-07 14:15:00 | 1755.08 | PARTIAL | 0.50 | 4.19% |
| SELL | retest2 | 2024-09-30 11:00:00 | 1854.80 | 2024-10-08 09:15:00 | 1783.10 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2024-09-30 11:30:00 | 1854.70 | 2024-10-08 09:15:00 | 1783.10 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2024-10-01 10:00:00 | 1854.85 | 2024-10-08 09:15:00 | 1783.10 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2024-10-01 14:30:00 | 1847.85 | 2024-10-08 09:15:00 | 1783.10 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1831.85 | 2024-10-08 09:15:00 | 1783.10 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2024-10-03 10:00:00 | 1847.45 | 2024-10-08 09:15:00 | 1750.80 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2024-10-03 10:00:00 | 1847.45 | 2024-10-08 09:15:00 | 1783.10 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2024-10-03 12:15:00 | 1842.95 | 2024-10-14 10:15:00 | 1794.85 | STOP_HIT | 1.00 | 2.61% |
| SELL | retest2 | 2024-10-10 11:30:00 | 1781.90 | 2024-10-14 10:15:00 | 1794.85 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-10-11 13:15:00 | 1784.55 | 2024-10-14 10:15:00 | 1794.85 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-10-11 14:00:00 | 1785.75 | 2024-10-14 10:15:00 | 1794.85 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-10-17 14:15:00 | 1859.10 | 2024-10-17 14:15:00 | 1835.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-10-18 10:15:00 | 1859.20 | 2024-10-21 13:15:00 | 1818.70 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-10-18 11:15:00 | 1859.30 | 2024-10-21 13:15:00 | 1818.70 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-10-18 13:15:00 | 1859.85 | 2024-10-21 13:15:00 | 1818.70 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-10-23 14:15:00 | 1807.95 | 2024-10-30 09:15:00 | 1717.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 10:30:00 | 1807.10 | 2024-10-30 09:15:00 | 1716.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 14:15:00 | 1807.95 | 2024-10-31 11:15:00 | 1627.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-24 10:30:00 | 1807.10 | 2024-10-31 13:15:00 | 1626.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-25 10:15:00 | 1671.35 | 2024-11-26 13:15:00 | 1675.90 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-11-25 11:15:00 | 1672.60 | 2024-12-02 09:15:00 | 1676.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-11-25 12:30:00 | 1672.95 | 2024-12-02 09:15:00 | 1676.55 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-11-25 13:30:00 | 1675.00 | 2024-12-02 09:15:00 | 1676.55 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2024-11-26 10:15:00 | 1645.50 | 2024-12-02 09:15:00 | 1676.55 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-11-27 12:45:00 | 1649.00 | 2024-12-02 09:15:00 | 1676.55 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-11-28 11:45:00 | 1646.50 | 2024-12-02 09:15:00 | 1676.55 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-11-29 10:00:00 | 1646.50 | 2024-12-02 09:15:00 | 1676.55 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-12-13 14:30:00 | 1791.45 | 2024-12-17 13:15:00 | 1767.15 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-12-17 10:00:00 | 1790.20 | 2024-12-17 13:15:00 | 1767.15 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-01-09 09:15:00 | 1746.65 | 2025-01-10 13:15:00 | 1659.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 1746.65 | 2025-01-14 09:15:00 | 1632.10 | STOP_HIT | 0.50 | 6.56% |
| SELL | retest2 | 2025-02-18 10:15:00 | 1216.35 | 2025-02-19 09:15:00 | 1268.15 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2025-02-21 12:45:00 | 1281.90 | 2025-02-21 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-02-24 11:30:00 | 1284.85 | 2025-03-04 10:15:00 | 1409.93 | TARGET_HIT | 1.00 | 9.73% |
| BUY | retest2 | 2025-02-25 09:15:00 | 1281.75 | 2025-03-04 13:15:00 | 1413.34 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2025-02-27 09:15:00 | 1302.10 | 2025-03-10 09:15:00 | 1432.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-27 14:45:00 | 1319.00 | 2025-03-12 12:15:00 | 1450.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-28 15:00:00 | 1318.90 | 2025-03-12 12:15:00 | 1450.79 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-24 14:45:00 | 1437.90 | 2025-03-28 11:15:00 | 1437.50 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-03-25 09:15:00 | 1440.30 | 2025-03-28 11:15:00 | 1437.50 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-03-25 10:30:00 | 1437.35 | 2025-03-28 11:15:00 | 1437.50 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-03-28 09:30:00 | 1439.30 | 2025-03-28 11:15:00 | 1437.50 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1298.55 | 2025-04-21 09:15:00 | 1311.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1297.50 | 2025-04-21 09:15:00 | 1311.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-04-15 09:30:00 | 1297.40 | 2025-04-21 09:15:00 | 1311.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-04-22 13:15:00 | 1345.90 | 2025-04-23 09:15:00 | 1302.00 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-04-29 10:30:00 | 1246.80 | 2025-04-29 11:15:00 | 1268.10 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1274.40 | 2025-06-23 10:15:00 | 1288.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-06-25 14:00:00 | 1310.00 | 2025-06-30 10:15:00 | 1302.60 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-06-27 15:15:00 | 1310.60 | 2025-06-30 10:15:00 | 1302.60 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-06-30 09:30:00 | 1315.90 | 2025-06-30 10:15:00 | 1302.60 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-04 14:30:00 | 1364.70 | 2025-07-07 09:15:00 | 1350.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-07 14:45:00 | 1363.00 | 2025-07-08 09:15:00 | 1353.20 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-08 11:30:00 | 1364.40 | 2025-07-11 10:15:00 | 1360.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-07-16 12:30:00 | 1391.20 | 2025-07-18 10:15:00 | 1381.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-16 13:00:00 | 1392.10 | 2025-07-18 10:15:00 | 1381.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-17 09:45:00 | 1391.10 | 2025-07-18 10:15:00 | 1381.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-17 13:15:00 | 1391.40 | 2025-07-18 10:15:00 | 1381.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-18 12:45:00 | 1386.50 | 2025-07-18 13:15:00 | 1377.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-22 09:15:00 | 1357.00 | 2025-07-30 14:15:00 | 1337.10 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2025-08-07 11:30:00 | 1306.30 | 2025-08-11 09:15:00 | 1240.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 11:30:00 | 1306.30 | 2025-08-12 12:15:00 | 1257.90 | STOP_HIT | 0.50 | 3.71% |
| BUY | retest2 | 2025-08-21 14:00:00 | 1366.00 | 2025-08-26 09:15:00 | 1348.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-21 15:15:00 | 1365.40 | 2025-08-26 09:15:00 | 1348.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-22 10:45:00 | 1364.90 | 2025-08-26 09:15:00 | 1348.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-25 09:30:00 | 1366.10 | 2025-08-26 09:15:00 | 1348.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-09-12 15:15:00 | 1398.00 | 2025-09-15 09:15:00 | 1409.60 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1357.60 | 2025-09-29 15:15:00 | 1381.90 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-09-29 12:00:00 | 1360.00 | 2025-09-29 15:15:00 | 1381.90 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-29 12:30:00 | 1359.20 | 2025-09-29 15:15:00 | 1381.90 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-29 13:15:00 | 1360.20 | 2025-09-29 15:15:00 | 1381.90 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-10-06 11:15:00 | 1348.50 | 2025-10-06 13:15:00 | 1357.70 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2025-10-10 09:15:00 | 1412.00 | 2025-10-13 09:15:00 | 1391.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-20 11:00:00 | 1423.30 | 2025-10-27 11:15:00 | 1432.80 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-10-27 09:45:00 | 1427.10 | 2025-10-27 11:15:00 | 1432.80 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1410.90 | 2025-11-06 11:15:00 | 1340.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1410.90 | 2025-11-07 11:15:00 | 1316.00 | STOP_HIT | 0.50 | 6.73% |
| BUY | retest2 | 2025-11-27 13:45:00 | 1386.70 | 2025-11-28 11:15:00 | 1365.70 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest1 | 2025-12-08 11:15:00 | 1318.10 | 2025-12-09 12:15:00 | 1336.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest1 | 2025-12-08 11:45:00 | 1318.10 | 2025-12-09 12:15:00 | 1336.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest1 | 2025-12-08 15:15:00 | 1318.00 | 2025-12-09 12:15:00 | 1336.10 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-11 10:15:00 | 1346.80 | 2025-12-19 13:15:00 | 1368.90 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-12-11 12:00:00 | 1351.50 | 2025-12-19 13:15:00 | 1368.90 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1446.40 | 2026-01-20 10:15:00 | 1374.65 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1447.00 | 2026-01-20 11:15:00 | 1374.08 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1445.50 | 2026-01-20 11:15:00 | 1373.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1446.40 | 2026-01-21 09:15:00 | 1301.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1447.00 | 2026-01-21 09:15:00 | 1302.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1445.50 | 2026-01-21 09:15:00 | 1300.95 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-17 15:00:00 | 1527.90 | 2026-02-19 14:15:00 | 1510.90 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-18 10:15:00 | 1528.00 | 2026-02-19 14:15:00 | 1510.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-02-18 11:45:00 | 1527.60 | 2026-02-19 14:15:00 | 1510.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-25 09:30:00 | 1540.20 | 2026-02-25 12:15:00 | 1522.90 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-03-06 11:30:00 | 1464.00 | 2026-03-11 10:15:00 | 1490.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1420.30 | 2026-03-11 10:15:00 | 1490.00 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1398.40 | 2026-03-18 10:15:00 | 1428.20 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1283.20 | 2026-04-01 14:15:00 | 1219.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:45:00 | 1263.90 | 2026-04-02 09:15:00 | 1200.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1283.20 | 2026-04-02 13:15:00 | 1231.30 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2026-04-01 10:45:00 | 1263.90 | 2026-04-02 13:15:00 | 1231.30 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2026-05-05 09:15:00 | 1424.80 | 2026-05-07 09:15:00 | 1353.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-05 09:15:00 | 1424.80 | 2026-05-07 11:15:00 | 1373.10 | STOP_HIT | 0.50 | 3.63% |
