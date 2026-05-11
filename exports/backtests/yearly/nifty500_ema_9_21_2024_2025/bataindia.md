# Bata India Ltd. (BATAINDIA)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 722.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 130 |
| ALERT1 | 86 |
| ALERT2 | 86 |
| ALERT2_SKIP | 43 |
| ALERT3 | 220 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 119 |
| PARTIAL | 16 |
| TARGET_HIT | 2 |
| STOP_HIT | 125 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 65 / 77
- **Target hits / Stop hits / Partials:** 2 / 124 / 16
- **Avg / median % per leg:** 0.83% / -0.17%
- **Sum % (uncompounded):** 117.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 22 | 40.0% | 1 | 54 | 0 | 0.24% | 13.3% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.93% | -1.9% |
| BUY @ 3rd Alert (retest2) | 53 | 22 | 41.5% | 1 | 52 | 0 | 0.29% | 15.1% |
| SELL (all) | 87 | 43 | 49.4% | 1 | 70 | 16 | 1.20% | 104.5% |
| SELL @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.91% | -4.5% |
| SELL @ 3rd Alert (retest2) | 82 | 43 | 52.4% | 1 | 65 | 16 | 1.33% | 109.0% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -0.91% | -6.4% |
| retest2 (combined) | 135 | 65 | 48.1% | 2 | 117 | 16 | 0.92% | 124.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 1321.75 | 1313.66 | 1312.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 1330.00 | 1326.39 | 1323.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 1327.00 | 1327.14 | 1324.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 1327.00 | 1327.14 | 1324.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 1322.90 | 1326.29 | 1324.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 1324.25 | 1326.29 | 1324.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1334.40 | 1327.91 | 1325.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 1343.90 | 1329.32 | 1326.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 13:15:00 | 1355.30 | 1358.19 | 1358.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 13:15:00 | 1355.30 | 1358.19 | 1358.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 1350.05 | 1355.78 | 1357.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 1359.90 | 1351.74 | 1353.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 10:15:00 | 1359.90 | 1351.74 | 1353.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 1359.90 | 1351.74 | 1353.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:45:00 | 1358.50 | 1351.74 | 1353.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 1360.00 | 1353.39 | 1354.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:45:00 | 1358.40 | 1353.39 | 1354.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 14:15:00 | 1355.20 | 1354.64 | 1354.62 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 15:15:00 | 1353.00 | 1354.31 | 1354.48 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 09:15:00 | 1357.70 | 1354.99 | 1354.77 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 1346.05 | 1353.83 | 1354.32 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 15:15:00 | 1357.60 | 1354.70 | 1354.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 09:15:00 | 1368.65 | 1357.49 | 1355.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 13:15:00 | 1387.45 | 1391.12 | 1377.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-30 14:00:00 | 1387.45 | 1391.12 | 1377.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1371.65 | 1387.22 | 1377.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:30:00 | 1371.30 | 1387.22 | 1377.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 1372.00 | 1384.18 | 1376.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 1382.35 | 1384.18 | 1376.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 14:15:00 | 1368.10 | 1380.57 | 1378.52 | SL hit (close<static) qty=1.00 sl=1369.05 alert=retest2 |

### Cycle 8 — SELL (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 15:15:00 | 1360.85 | 1376.63 | 1376.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 09:15:00 | 1345.75 | 1366.92 | 1371.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 1349.75 | 1345.06 | 1357.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 13:45:00 | 1353.00 | 1345.06 | 1357.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1410.05 | 1356.71 | 1359.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 1410.05 | 1356.71 | 1359.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 1424.00 | 1370.17 | 1365.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 1461.05 | 1419.68 | 1397.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 1480.00 | 1480.82 | 1463.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 1489.35 | 1480.82 | 1463.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 13:30:00 | 1486.40 | 1485.42 | 1473.03 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 1475.05 | 1481.80 | 1473.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1476.35 | 1481.80 | 1473.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 10:15:00 | 1474.05 | 1480.10 | 1474.10 | SL hit (close<ema400) qty=1.00 sl=1474.10 alert=retest1 |

### Cycle 10 — SELL (started 2024-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 15:15:00 | 1458.00 | 1469.73 | 1470.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 14:15:00 | 1452.55 | 1461.01 | 1464.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 14:15:00 | 1457.40 | 1455.93 | 1459.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 14:15:00 | 1457.40 | 1455.93 | 1459.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 1457.40 | 1455.93 | 1459.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 15:00:00 | 1457.40 | 1455.93 | 1459.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 1459.00 | 1456.54 | 1459.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 1456.00 | 1456.54 | 1459.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1449.80 | 1455.19 | 1458.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:15:00 | 1448.45 | 1455.19 | 1458.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 15:00:00 | 1448.65 | 1454.55 | 1457.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 11:15:00 | 1464.60 | 1457.89 | 1457.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 1464.60 | 1457.89 | 1457.87 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 1454.00 | 1460.42 | 1460.82 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 10:15:00 | 1471.40 | 1462.65 | 1461.70 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 1459.40 | 1461.62 | 1461.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 09:15:00 | 1449.80 | 1459.26 | 1460.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 15:15:00 | 1446.90 | 1445.14 | 1449.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 09:15:00 | 1456.45 | 1445.14 | 1449.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1460.35 | 1448.18 | 1450.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:00:00 | 1460.35 | 1448.18 | 1450.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 1457.45 | 1450.04 | 1451.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:45:00 | 1459.50 | 1450.04 | 1451.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 11:15:00 | 1478.70 | 1455.77 | 1453.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 12:15:00 | 1495.50 | 1463.72 | 1457.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 13:15:00 | 1491.50 | 1493.89 | 1481.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 13:45:00 | 1493.70 | 1493.89 | 1481.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 1490.40 | 1493.42 | 1486.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 1490.40 | 1493.42 | 1486.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 1491.50 | 1496.23 | 1491.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:00:00 | 1491.50 | 1496.23 | 1491.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 1492.70 | 1495.53 | 1491.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:45:00 | 1490.90 | 1495.53 | 1491.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 1490.00 | 1494.42 | 1491.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:30:00 | 1491.10 | 1494.42 | 1491.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 1492.55 | 1494.05 | 1491.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:45:00 | 1497.00 | 1494.26 | 1491.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 1496.30 | 1493.78 | 1491.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:45:00 | 1495.00 | 1494.39 | 1492.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:15:00 | 1497.60 | 1494.23 | 1492.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1497.60 | 1504.59 | 1501.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 1497.60 | 1504.59 | 1501.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1498.70 | 1503.41 | 1501.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:30:00 | 1491.05 | 1503.41 | 1501.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 1507.75 | 1505.01 | 1502.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:30:00 | 1503.10 | 1505.01 | 1502.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 1517.00 | 1518.16 | 1513.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 1514.85 | 1518.16 | 1513.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1515.40 | 1517.61 | 1513.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:45:00 | 1520.25 | 1517.61 | 1513.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1507.70 | 1515.63 | 1513.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 1499.00 | 1515.63 | 1513.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1523.05 | 1517.11 | 1514.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:30:00 | 1526.30 | 1520.46 | 1516.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 11:45:00 | 1527.00 | 1532.29 | 1528.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 15:15:00 | 1514.00 | 1524.20 | 1525.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 1514.00 | 1524.20 | 1525.38 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 1535.00 | 1526.07 | 1526.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 09:15:00 | 1543.20 | 1533.17 | 1530.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 14:15:00 | 1530.75 | 1536.42 | 1533.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 14:15:00 | 1530.75 | 1536.42 | 1533.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1530.75 | 1536.42 | 1533.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 1530.75 | 1536.42 | 1533.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 1527.70 | 1534.68 | 1533.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 1520.50 | 1534.68 | 1533.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 1526.85 | 1532.52 | 1532.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 1526.85 | 1532.52 | 1532.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 1528.15 | 1531.65 | 1531.95 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 1531.40 | 1530.39 | 1530.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 1548.80 | 1538.32 | 1534.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 13:15:00 | 1597.55 | 1599.11 | 1582.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 13:30:00 | 1594.90 | 1599.11 | 1582.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1592.85 | 1613.68 | 1606.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:30:00 | 1592.00 | 1613.68 | 1606.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1598.10 | 1610.56 | 1605.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:15:00 | 1602.75 | 1610.56 | 1605.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 10:45:00 | 1604.55 | 1604.94 | 1604.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 11:15:00 | 1600.00 | 1603.95 | 1604.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 11:15:00 | 1600.00 | 1603.95 | 1604.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 1596.40 | 1601.50 | 1603.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 1610.00 | 1602.14 | 1603.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 1610.00 | 1602.14 | 1603.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1610.00 | 1602.14 | 1603.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:45:00 | 1607.50 | 1602.14 | 1603.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 10:15:00 | 1609.90 | 1603.69 | 1603.69 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 1603.35 | 1603.62 | 1603.66 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 12:15:00 | 1611.95 | 1605.29 | 1604.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 13:15:00 | 1617.75 | 1607.78 | 1605.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 1606.25 | 1616.05 | 1613.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 1606.25 | 1616.05 | 1613.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1606.25 | 1616.05 | 1613.24 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 1588.00 | 1610.44 | 1610.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 1557.75 | 1596.35 | 1604.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 11:15:00 | 1402.10 | 1399.11 | 1409.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 11:30:00 | 1404.35 | 1399.11 | 1409.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 1420.00 | 1403.29 | 1410.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:30:00 | 1419.90 | 1403.29 | 1410.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 1421.05 | 1406.84 | 1411.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:30:00 | 1421.65 | 1406.84 | 1411.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 1426.60 | 1415.99 | 1414.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 1429.30 | 1418.65 | 1416.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 1422.90 | 1424.81 | 1421.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 1422.90 | 1424.81 | 1421.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 1422.90 | 1424.81 | 1421.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:45:00 | 1422.45 | 1424.81 | 1421.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 1418.20 | 1423.49 | 1420.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 1418.20 | 1423.49 | 1420.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1413.45 | 1421.48 | 1420.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 1412.00 | 1421.48 | 1420.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 1418.60 | 1420.65 | 1419.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:45:00 | 1417.95 | 1420.65 | 1419.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 1418.90 | 1420.30 | 1419.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:30:00 | 1419.85 | 1420.30 | 1419.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 1418.90 | 1420.02 | 1419.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 1423.85 | 1420.02 | 1419.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 1417.45 | 1419.51 | 1419.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 09:15:00 | 1417.45 | 1419.51 | 1419.55 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 11:15:00 | 1421.50 | 1419.62 | 1419.58 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 12:15:00 | 1418.85 | 1419.47 | 1419.51 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 14:15:00 | 1420.70 | 1419.66 | 1419.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 1438.75 | 1423.67 | 1421.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1436.85 | 1437.99 | 1431.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 1436.85 | 1437.99 | 1431.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1436.85 | 1437.99 | 1431.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 1437.00 | 1437.99 | 1431.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 1446.95 | 1449.33 | 1445.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:00:00 | 1446.95 | 1449.33 | 1445.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1450.00 | 1449.47 | 1445.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 13:30:00 | 1451.45 | 1449.14 | 1446.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 14:30:00 | 1450.05 | 1449.96 | 1446.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 15:15:00 | 1444.05 | 1454.47 | 1452.10 | SL hit (close<static) qty=1.00 sl=1445.55 alert=retest2 |

### Cycle 30 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 1439.05 | 1449.87 | 1450.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 1436.05 | 1447.11 | 1449.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 1459.40 | 1449.05 | 1449.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 15:15:00 | 1459.40 | 1449.05 | 1449.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1459.40 | 1449.05 | 1449.79 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 1454.10 | 1450.29 | 1450.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 13:15:00 | 1457.35 | 1451.70 | 1450.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 14:15:00 | 1451.55 | 1451.67 | 1450.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 14:15:00 | 1451.55 | 1451.67 | 1450.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1451.55 | 1451.67 | 1450.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 1451.55 | 1451.67 | 1450.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1465.50 | 1465.85 | 1461.82 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 13:15:00 | 1452.25 | 1460.13 | 1460.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 14:15:00 | 1451.85 | 1458.47 | 1459.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 1417.40 | 1415.95 | 1429.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 11:00:00 | 1417.40 | 1415.95 | 1429.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1428.35 | 1418.56 | 1426.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 1428.35 | 1418.56 | 1426.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1430.00 | 1420.85 | 1426.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 1435.30 | 1420.85 | 1426.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 1438.00 | 1431.28 | 1430.41 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 1417.00 | 1428.46 | 1429.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 1416.90 | 1426.15 | 1428.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 11:15:00 | 1428.40 | 1424.76 | 1426.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 11:15:00 | 1428.40 | 1424.76 | 1426.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 1428.40 | 1424.76 | 1426.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:00:00 | 1428.40 | 1424.76 | 1426.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 1422.10 | 1424.23 | 1426.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:45:00 | 1421.45 | 1424.17 | 1426.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 1434.25 | 1427.77 | 1427.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 1434.25 | 1427.77 | 1427.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 11:15:00 | 1441.80 | 1432.18 | 1429.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 1439.20 | 1440.67 | 1437.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 09:45:00 | 1438.85 | 1440.67 | 1437.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 1437.90 | 1440.12 | 1437.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 1439.00 | 1440.12 | 1437.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 1439.50 | 1439.65 | 1437.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:15:00 | 1438.85 | 1439.65 | 1437.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 1440.00 | 1439.72 | 1438.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 1443.65 | 1439.72 | 1438.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 11:15:00 | 1430.80 | 1438.35 | 1438.16 | SL hit (close<static) qty=1.00 sl=1437.05 alert=retest2 |

### Cycle 36 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 1423.20 | 1435.32 | 1436.80 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 1431.00 | 1428.14 | 1427.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 1434.00 | 1430.04 | 1429.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 11:15:00 | 1426.80 | 1429.83 | 1429.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 11:15:00 | 1426.80 | 1429.83 | 1429.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 1426.80 | 1429.83 | 1429.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:45:00 | 1427.10 | 1429.83 | 1429.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 1430.05 | 1429.87 | 1429.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 13:15:00 | 1432.80 | 1429.87 | 1429.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 1423.40 | 1428.25 | 1428.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 1423.40 | 1428.25 | 1428.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 10:15:00 | 1418.00 | 1426.20 | 1427.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 1426.45 | 1420.49 | 1423.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 14:15:00 | 1426.45 | 1420.49 | 1423.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 1426.45 | 1420.49 | 1423.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 1426.45 | 1420.49 | 1423.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1429.95 | 1422.38 | 1424.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 1419.10 | 1422.38 | 1424.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 11:00:00 | 1419.10 | 1421.18 | 1423.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 11:45:00 | 1420.45 | 1421.79 | 1423.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:00:00 | 1418.85 | 1421.20 | 1423.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1427.05 | 1422.17 | 1423.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 1427.05 | 1422.17 | 1423.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1425.00 | 1422.74 | 1423.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 1438.90 | 1422.74 | 1423.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 1446.30 | 1427.45 | 1425.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 1446.30 | 1427.45 | 1425.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 15:15:00 | 1454.00 | 1441.81 | 1434.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 11:15:00 | 1445.30 | 1445.53 | 1438.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 12:00:00 | 1445.30 | 1445.53 | 1438.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 1440.30 | 1444.08 | 1438.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 1440.30 | 1444.08 | 1438.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 1440.60 | 1443.39 | 1439.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:30:00 | 1436.65 | 1443.39 | 1439.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 1437.75 | 1442.26 | 1438.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 1441.90 | 1442.26 | 1438.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 11:15:00 | 1435.20 | 1439.49 | 1438.41 | SL hit (close<static) qty=1.00 sl=1435.25 alert=retest2 |

### Cycle 40 — SELL (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 13:15:00 | 1433.65 | 1437.40 | 1437.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 14:15:00 | 1427.70 | 1435.46 | 1436.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 1380.50 | 1362.45 | 1373.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 1380.50 | 1362.45 | 1373.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1380.50 | 1362.45 | 1373.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 1380.50 | 1362.45 | 1373.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 1385.50 | 1367.06 | 1374.28 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 1389.10 | 1377.82 | 1377.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 1390.30 | 1380.32 | 1378.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 1401.70 | 1402.07 | 1394.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 1401.70 | 1402.07 | 1394.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 1395.00 | 1400.51 | 1395.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 1395.00 | 1400.51 | 1395.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 1393.00 | 1399.01 | 1395.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 10:15:00 | 1400.00 | 1398.22 | 1395.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 11:00:00 | 1402.30 | 1399.03 | 1395.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:30:00 | 1401.65 | 1403.26 | 1400.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 13:15:00 | 1429.00 | 1438.45 | 1438.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 13:15:00 | 1429.00 | 1438.45 | 1438.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 1419.00 | 1432.49 | 1435.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 11:15:00 | 1406.90 | 1404.05 | 1415.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 12:00:00 | 1406.90 | 1404.05 | 1415.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1415.70 | 1406.38 | 1415.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 1415.70 | 1406.38 | 1415.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 1408.95 | 1406.89 | 1414.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 1416.00 | 1406.89 | 1414.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1365.70 | 1383.01 | 1394.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:30:00 | 1363.35 | 1380.66 | 1392.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:30:00 | 1364.00 | 1374.18 | 1386.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:00:00 | 1363.00 | 1374.18 | 1386.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 15:00:00 | 1363.75 | 1372.09 | 1384.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 1382.40 | 1373.14 | 1381.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 1382.40 | 1373.14 | 1381.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1381.00 | 1374.72 | 1381.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 1379.05 | 1374.72 | 1381.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 1377.20 | 1375.21 | 1381.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 15:15:00 | 1370.05 | 1376.10 | 1380.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 10:45:00 | 1371.30 | 1365.68 | 1369.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 11:15:00 | 1372.55 | 1365.68 | 1369.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 13:45:00 | 1370.40 | 1369.49 | 1370.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1366.95 | 1367.65 | 1369.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:00:00 | 1366.95 | 1367.65 | 1369.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1347.25 | 1357.11 | 1363.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:30:00 | 1365.95 | 1357.11 | 1363.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1362.00 | 1358.07 | 1362.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1362.00 | 1358.07 | 1362.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1357.45 | 1357.95 | 1362.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 1345.60 | 1357.95 | 1362.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:45:00 | 1350.65 | 1355.76 | 1360.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:15:00 | 1301.55 | 1337.41 | 1348.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:15:00 | 1302.73 | 1337.41 | 1348.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:15:00 | 1303.92 | 1337.41 | 1348.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:15:00 | 1301.88 | 1337.41 | 1348.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-05 10:15:00 | 1340.00 | 1337.93 | 1347.37 | SL hit (close>ema200) qty=0.50 sl=1337.93 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 1355.00 | 1349.62 | 1349.02 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 1338.45 | 1350.10 | 1350.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1331.80 | 1336.78 | 1340.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 1334.95 | 1332.72 | 1335.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 10:15:00 | 1334.95 | 1332.72 | 1335.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1334.95 | 1332.72 | 1335.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 1334.95 | 1332.72 | 1335.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 1346.85 | 1335.55 | 1336.92 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 13:15:00 | 1343.60 | 1338.76 | 1338.24 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 1331.35 | 1336.71 | 1337.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 1316.95 | 1329.03 | 1333.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1333.00 | 1328.27 | 1331.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1333.00 | 1328.27 | 1331.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1333.00 | 1328.27 | 1331.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1333.00 | 1328.27 | 1331.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1328.15 | 1328.24 | 1331.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 1333.00 | 1328.24 | 1331.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1334.90 | 1329.58 | 1331.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:00:00 | 1334.90 | 1329.58 | 1331.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1330.30 | 1329.72 | 1331.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:30:00 | 1325.85 | 1328.09 | 1330.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 15:15:00 | 1299.30 | 1293.88 | 1293.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 1299.30 | 1293.88 | 1293.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 1322.15 | 1299.54 | 1296.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 1363.15 | 1367.58 | 1355.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 11:00:00 | 1363.15 | 1367.58 | 1355.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 1362.65 | 1365.82 | 1357.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 1362.65 | 1365.82 | 1357.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1361.15 | 1364.88 | 1358.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 1361.15 | 1364.88 | 1358.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 1358.05 | 1363.52 | 1358.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 1377.00 | 1363.52 | 1358.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 1379.80 | 1365.58 | 1359.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 13:15:00 | 1432.00 | 1443.37 | 1443.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 1432.00 | 1443.37 | 1443.55 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 12:15:00 | 1448.90 | 1444.02 | 1443.36 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 1431.95 | 1443.22 | 1443.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 1422.25 | 1437.21 | 1440.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 15:15:00 | 1432.00 | 1430.95 | 1435.83 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:15:00 | 1424.05 | 1430.95 | 1435.83 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 1431.75 | 1427.54 | 1432.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 1431.75 | 1427.54 | 1432.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 1428.40 | 1427.71 | 1432.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 1428.40 | 1427.71 | 1432.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 1431.50 | 1428.47 | 1432.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 1431.50 | 1428.47 | 1432.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1432.75 | 1429.33 | 1432.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-13 14:15:00 | 1432.75 | 1429.33 | 1432.26 | SL hit (close>ema400) qty=1.00 sl=1432.26 alert=retest1 |

### Cycle 51 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 1437.30 | 1433.26 | 1433.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 1437.80 | 1434.16 | 1433.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 1428.65 | 1433.06 | 1433.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 10:15:00 | 1428.65 | 1433.06 | 1433.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1428.65 | 1433.06 | 1433.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:00:00 | 1428.65 | 1433.06 | 1433.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 1422.75 | 1431.00 | 1432.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 1421.70 | 1429.14 | 1431.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1343.60 | 1341.84 | 1354.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 1343.60 | 1341.84 | 1354.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 1343.95 | 1343.16 | 1352.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:30:00 | 1349.95 | 1343.16 | 1352.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 1360.95 | 1348.22 | 1352.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 14:45:00 | 1360.00 | 1348.22 | 1352.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 1362.15 | 1351.00 | 1353.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 1353.75 | 1351.00 | 1353.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:15:00 | 1359.00 | 1353.77 | 1354.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 1376.60 | 1355.87 | 1354.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1376.60 | 1355.87 | 1354.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 10:15:00 | 1408.40 | 1366.37 | 1359.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 1372.20 | 1377.99 | 1370.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 1372.20 | 1377.99 | 1370.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1372.20 | 1377.99 | 1370.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 1363.55 | 1377.99 | 1370.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 1384.95 | 1379.38 | 1371.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 15:00:00 | 1389.15 | 1379.70 | 1374.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:00:00 | 1391.60 | 1381.06 | 1378.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 09:15:00 | 1408.45 | 1424.15 | 1425.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 1408.45 | 1424.15 | 1425.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 1397.10 | 1418.74 | 1423.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 1412.85 | 1409.64 | 1417.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 1412.85 | 1409.64 | 1417.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1424.00 | 1412.51 | 1417.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 1424.00 | 1412.51 | 1417.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1419.75 | 1413.96 | 1418.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 1421.90 | 1413.96 | 1418.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1428.65 | 1416.90 | 1418.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 1428.65 | 1416.90 | 1418.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1426.55 | 1418.83 | 1419.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:30:00 | 1430.65 | 1418.83 | 1419.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 1420.70 | 1417.89 | 1418.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 1414.25 | 1417.89 | 1418.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1416.60 | 1417.63 | 1418.65 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 10:15:00 | 1430.00 | 1420.11 | 1419.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-10 11:15:00 | 1435.00 | 1423.08 | 1421.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 14:15:00 | 1425.75 | 1427.18 | 1423.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 14:15:00 | 1425.75 | 1427.18 | 1423.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 1425.75 | 1427.18 | 1423.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 15:00:00 | 1425.75 | 1427.18 | 1423.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 09:15:00 | 1388.35 | 1419.04 | 1420.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 10:15:00 | 1372.30 | 1409.69 | 1416.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 1314.50 | 1311.58 | 1325.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 1302.95 | 1309.81 | 1317.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1302.95 | 1309.81 | 1317.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 1293.10 | 1306.31 | 1311.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:00:00 | 1298.10 | 1302.79 | 1309.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 1298.05 | 1291.18 | 1292.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 14:15:00 | 1299.60 | 1292.86 | 1292.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 1299.60 | 1292.86 | 1292.82 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 1283.00 | 1292.20 | 1292.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 1281.35 | 1290.03 | 1291.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1276.00 | 1249.84 | 1257.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 1276.00 | 1249.84 | 1257.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1276.00 | 1249.84 | 1257.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1276.00 | 1249.84 | 1257.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1265.55 | 1252.98 | 1258.22 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 1263.35 | 1260.88 | 1260.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1273.00 | 1263.30 | 1261.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 12:15:00 | 1262.75 | 1266.76 | 1264.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 12:15:00 | 1262.75 | 1266.76 | 1264.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 1262.75 | 1266.76 | 1264.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:00:00 | 1262.75 | 1266.76 | 1264.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1260.00 | 1265.41 | 1263.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 1260.00 | 1265.41 | 1263.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1269.20 | 1266.17 | 1264.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 1267.00 | 1266.17 | 1264.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 1274.90 | 1267.92 | 1265.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 1290.40 | 1267.92 | 1265.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-03 13:15:00 | 1419.44 | 1382.76 | 1345.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 1356.25 | 1365.09 | 1365.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 1344.00 | 1359.58 | 1362.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 12:15:00 | 1344.70 | 1341.42 | 1347.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 12:15:00 | 1344.70 | 1341.42 | 1347.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 1344.70 | 1341.42 | 1347.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:45:00 | 1345.25 | 1341.42 | 1347.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 1345.00 | 1341.45 | 1345.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:00:00 | 1327.70 | 1337.95 | 1343.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 11:15:00 | 1377.55 | 1350.51 | 1347.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 11:15:00 | 1377.55 | 1350.51 | 1347.08 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 10:15:00 | 1326.10 | 1344.86 | 1346.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 1318.00 | 1330.24 | 1336.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 15:15:00 | 1299.00 | 1298.13 | 1311.06 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:15:00 | 1282.05 | 1298.13 | 1311.06 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 14:15:00 | 1283.00 | 1284.15 | 1297.75 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1274.55 | 1282.04 | 1293.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:45:00 | 1262.55 | 1278.90 | 1290.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 1289.45 | 1267.63 | 1271.58 | SL hit (close>ema400) qty=1.00 sl=1271.58 alert=retest1 |

### Cycle 63 — BUY (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 10:15:00 | 1307.95 | 1275.69 | 1274.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 11:15:00 | 1318.65 | 1284.28 | 1278.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 13:15:00 | 1342.50 | 1347.28 | 1327.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 13:45:00 | 1343.65 | 1347.28 | 1327.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 1322.60 | 1341.75 | 1330.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 1322.60 | 1341.75 | 1330.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 1308.55 | 1335.11 | 1328.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:00:00 | 1308.55 | 1335.11 | 1328.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 13:15:00 | 1296.95 | 1318.71 | 1321.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 14:15:00 | 1289.80 | 1312.93 | 1318.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 1234.60 | 1216.82 | 1244.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 15:00:00 | 1234.60 | 1216.82 | 1244.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 1257.00 | 1224.85 | 1245.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 1212.40 | 1224.85 | 1245.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 11:00:00 | 1230.65 | 1221.22 | 1230.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 12:00:00 | 1229.85 | 1222.94 | 1230.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 13:30:00 | 1230.55 | 1226.20 | 1230.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 1238.90 | 1228.74 | 1231.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 15:00:00 | 1238.90 | 1228.74 | 1231.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 1239.00 | 1230.79 | 1232.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:15:00 | 1245.80 | 1230.79 | 1232.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 1250.35 | 1234.70 | 1234.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 1250.35 | 1234.70 | 1234.00 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 15:15:00 | 1230.85 | 1240.68 | 1241.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 09:15:00 | 1224.10 | 1237.37 | 1239.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 1219.85 | 1207.66 | 1215.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 1219.85 | 1207.66 | 1215.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1219.85 | 1207.66 | 1215.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 1228.45 | 1207.66 | 1215.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 1214.85 | 1209.10 | 1215.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 1208.45 | 1209.10 | 1215.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 12:45:00 | 1211.45 | 1209.01 | 1214.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 14:15:00 | 1229.85 | 1214.20 | 1216.01 | SL hit (close>static) qty=1.00 sl=1226.40 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 1235.10 | 1219.85 | 1218.27 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 14:15:00 | 1209.65 | 1220.23 | 1221.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 15:15:00 | 1207.00 | 1217.58 | 1220.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 1227.85 | 1219.64 | 1220.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 1227.85 | 1219.64 | 1220.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1227.85 | 1219.64 | 1220.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 1227.85 | 1219.64 | 1220.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 1234.15 | 1222.54 | 1222.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 15:15:00 | 1238.00 | 1231.42 | 1227.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 09:15:00 | 1263.60 | 1266.48 | 1256.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 10:00:00 | 1263.60 | 1266.48 | 1256.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 1263.00 | 1265.26 | 1257.64 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 09:15:00 | 1251.80 | 1254.46 | 1254.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-24 13:15:00 | 1237.65 | 1248.93 | 1251.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 1234.30 | 1234.11 | 1240.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 1234.30 | 1234.11 | 1240.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1234.30 | 1234.11 | 1240.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:00:00 | 1227.05 | 1232.70 | 1239.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:30:00 | 1226.10 | 1231.03 | 1237.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 09:15:00 | 1237.80 | 1220.46 | 1220.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 1237.80 | 1220.46 | 1220.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 1242.45 | 1232.71 | 1228.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 1230.35 | 1235.62 | 1230.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 1230.35 | 1235.62 | 1230.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1230.35 | 1235.62 | 1230.93 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 15:15:00 | 1220.65 | 1228.82 | 1229.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 1208.65 | 1224.79 | 1227.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 15:15:00 | 1211.00 | 1210.14 | 1217.34 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-07 09:15:00 | 1178.00 | 1210.14 | 1217.34 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1210.45 | 1187.41 | 1196.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 1210.45 | 1187.41 | 1196.52 | SL hit (close>ema400) qty=1.00 sl=1196.52 alert=retest1 |

### Cycle 73 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 1223.75 | 1205.57 | 1203.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 1240.00 | 1226.51 | 1217.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 10:15:00 | 1232.70 | 1236.33 | 1228.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 11:00:00 | 1232.70 | 1236.33 | 1228.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 1232.50 | 1235.57 | 1228.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:30:00 | 1229.80 | 1235.57 | 1228.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 13:15:00 | 1230.00 | 1233.75 | 1229.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 14:00:00 | 1230.00 | 1233.75 | 1229.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 14:15:00 | 1229.60 | 1232.92 | 1229.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 14:30:00 | 1231.40 | 1232.92 | 1229.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 15:15:00 | 1231.70 | 1232.68 | 1229.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 09:15:00 | 1234.00 | 1232.68 | 1229.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 11:00:00 | 1233.60 | 1232.67 | 1230.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 12:00:00 | 1232.70 | 1232.68 | 1230.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 12:30:00 | 1234.00 | 1232.44 | 1230.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 1239.20 | 1233.79 | 1231.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-17 12:15:00 | 1223.00 | 1229.99 | 1230.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 12:15:00 | 1223.00 | 1229.99 | 1230.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 12:15:00 | 1213.00 | 1224.22 | 1227.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-22 09:15:00 | 1216.20 | 1215.15 | 1221.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-22 10:00:00 | 1216.20 | 1215.15 | 1221.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 1221.00 | 1216.32 | 1221.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 11:00:00 | 1221.00 | 1216.32 | 1221.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 11:15:00 | 1231.10 | 1219.27 | 1222.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 12:00:00 | 1231.10 | 1219.27 | 1222.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 12:15:00 | 1223.80 | 1220.18 | 1222.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-23 10:00:00 | 1218.10 | 1221.73 | 1222.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 1229.00 | 1223.19 | 1223.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 10:15:00 | 1229.00 | 1223.19 | 1223.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 1232.20 | 1227.39 | 1225.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 09:15:00 | 1228.00 | 1228.28 | 1226.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 1228.00 | 1228.28 | 1226.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1228.00 | 1228.28 | 1226.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 1225.50 | 1228.28 | 1226.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1225.90 | 1227.80 | 1226.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:00:00 | 1225.90 | 1227.80 | 1226.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 1221.00 | 1226.44 | 1225.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 14:15:00 | 1228.20 | 1225.97 | 1225.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1212.90 | 1223.01 | 1224.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1212.90 | 1223.01 | 1224.27 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 1224.20 | 1220.14 | 1220.00 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 13:15:00 | 1214.20 | 1218.84 | 1219.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 1211.20 | 1217.32 | 1218.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 12:15:00 | 1210.40 | 1210.10 | 1214.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-30 12:30:00 | 1210.40 | 1210.10 | 1214.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1207.80 | 1208.19 | 1211.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:15:00 | 1194.70 | 1204.95 | 1209.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:45:00 | 1193.70 | 1202.66 | 1208.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 13:30:00 | 1194.30 | 1201.83 | 1207.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 10:15:00 | 1213.50 | 1207.62 | 1206.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 10:15:00 | 1213.50 | 1207.62 | 1206.93 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 1199.00 | 1205.91 | 1206.45 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 1213.70 | 1207.88 | 1207.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 1224.40 | 1212.00 | 1209.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 1207.00 | 1214.31 | 1211.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 1207.00 | 1214.31 | 1211.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1207.00 | 1214.31 | 1211.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 1207.00 | 1214.31 | 1211.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1201.80 | 1211.81 | 1210.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 1201.80 | 1211.81 | 1210.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1200.00 | 1209.45 | 1209.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 1193.80 | 1206.32 | 1208.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1211.60 | 1200.17 | 1202.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1211.60 | 1200.17 | 1202.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1211.60 | 1200.17 | 1202.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:45:00 | 1209.10 | 1201.97 | 1203.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 12:45:00 | 1208.40 | 1204.16 | 1204.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 13:15:00 | 1211.80 | 1205.69 | 1204.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1211.80 | 1205.69 | 1204.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1214.20 | 1207.39 | 1205.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 1209.90 | 1212.00 | 1208.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 11:15:00 | 1209.90 | 1212.00 | 1208.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 1209.90 | 1212.00 | 1208.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:45:00 | 1211.80 | 1212.00 | 1208.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1208.00 | 1211.20 | 1208.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 1206.10 | 1211.20 | 1208.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1205.00 | 1209.96 | 1208.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 1205.00 | 1209.96 | 1208.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1202.50 | 1208.47 | 1207.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:30:00 | 1204.80 | 1208.47 | 1207.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 15:15:00 | 1203.10 | 1207.39 | 1207.47 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 1209.60 | 1207.83 | 1207.67 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 1205.20 | 1207.56 | 1207.59 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 1211.60 | 1208.37 | 1207.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 1220.00 | 1214.02 | 1211.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 12:15:00 | 1248.00 | 1248.02 | 1238.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:00:00 | 1248.00 | 1248.02 | 1238.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 1242.10 | 1246.55 | 1240.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 1238.60 | 1246.55 | 1240.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1242.00 | 1245.64 | 1240.50 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1224.00 | 1235.81 | 1237.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1216.00 | 1231.85 | 1235.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1241.90 | 1231.43 | 1234.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1241.90 | 1231.43 | 1234.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1241.90 | 1231.43 | 1234.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1241.90 | 1231.43 | 1234.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1242.70 | 1233.69 | 1234.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:15:00 | 1247.50 | 1233.69 | 1234.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 1253.80 | 1237.71 | 1236.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 1261.20 | 1255.31 | 1249.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1270.20 | 1278.66 | 1271.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1270.20 | 1278.66 | 1271.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1270.20 | 1278.66 | 1271.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 1268.30 | 1278.66 | 1271.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1280.30 | 1278.99 | 1272.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:30:00 | 1285.00 | 1279.59 | 1273.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1263.50 | 1276.37 | 1274.30 | SL hit (close<static) qty=1.00 sl=1268.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 1265.00 | 1273.50 | 1274.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 11:15:00 | 1261.80 | 1271.16 | 1273.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 10:15:00 | 1264.50 | 1262.66 | 1267.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:30:00 | 1263.90 | 1262.66 | 1267.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1265.10 | 1264.28 | 1266.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:45:00 | 1268.20 | 1264.28 | 1266.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1266.00 | 1264.63 | 1266.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 1266.00 | 1264.63 | 1266.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1260.00 | 1263.70 | 1265.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:30:00 | 1254.80 | 1261.66 | 1264.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 1255.90 | 1261.02 | 1263.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1223.90 | 1220.46 | 1220.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 10:15:00 | 1223.90 | 1220.46 | 1220.25 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1213.00 | 1219.46 | 1219.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1210.00 | 1214.94 | 1217.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 1212.90 | 1211.69 | 1213.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 1212.90 | 1211.69 | 1213.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1217.30 | 1212.81 | 1214.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1217.30 | 1212.81 | 1214.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1216.00 | 1213.45 | 1214.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1215.60 | 1213.45 | 1214.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 1222.00 | 1216.37 | 1215.63 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1209.10 | 1215.20 | 1215.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1207.50 | 1210.58 | 1212.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 1206.30 | 1206.18 | 1208.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 1206.30 | 1206.18 | 1208.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1206.30 | 1206.18 | 1208.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1206.30 | 1206.18 | 1208.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1200.20 | 1204.99 | 1207.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:00:00 | 1197.80 | 1201.25 | 1204.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 15:00:00 | 1196.30 | 1199.58 | 1203.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 1211.20 | 1203.59 | 1203.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 1211.20 | 1203.59 | 1203.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 15:15:00 | 1215.00 | 1210.33 | 1207.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 1210.40 | 1210.55 | 1207.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 1210.40 | 1210.55 | 1207.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1205.80 | 1209.60 | 1207.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:45:00 | 1205.90 | 1209.60 | 1207.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 1207.10 | 1209.10 | 1207.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:15:00 | 1223.60 | 1208.88 | 1207.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 1225.00 | 1234.26 | 1234.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 1225.00 | 1234.26 | 1234.33 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 1259.90 | 1239.39 | 1236.65 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 1245.00 | 1255.17 | 1255.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 1237.20 | 1251.58 | 1253.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1246.90 | 1240.71 | 1244.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1246.90 | 1240.71 | 1244.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1246.90 | 1240.71 | 1244.02 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1258.00 | 1248.13 | 1247.07 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 1244.50 | 1248.87 | 1249.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 1235.00 | 1245.16 | 1247.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 15:15:00 | 1210.80 | 1210.72 | 1215.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:15:00 | 1207.10 | 1210.72 | 1215.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1210.50 | 1207.10 | 1210.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:45:00 | 1212.00 | 1207.10 | 1210.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1211.60 | 1208.00 | 1210.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 1211.00 | 1208.00 | 1210.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1208.60 | 1208.12 | 1210.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:00:00 | 1205.60 | 1209.06 | 1210.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 15:15:00 | 1203.90 | 1207.64 | 1208.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 1186.50 | 1183.44 | 1183.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 1186.50 | 1183.44 | 1183.30 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 1163.50 | 1180.68 | 1182.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1149.00 | 1166.55 | 1174.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1115.70 | 1077.91 | 1100.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1115.70 | 1077.91 | 1100.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1115.70 | 1077.91 | 1100.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1115.70 | 1077.91 | 1100.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1120.90 | 1086.51 | 1102.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 1122.00 | 1086.51 | 1102.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 1113.30 | 1109.07 | 1108.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 1131.70 | 1116.38 | 1112.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 1130.00 | 1131.14 | 1122.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:00:00 | 1130.00 | 1131.14 | 1122.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1122.00 | 1128.32 | 1124.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1123.60 | 1128.32 | 1124.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1123.10 | 1127.28 | 1124.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:15:00 | 1120.90 | 1127.28 | 1124.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1122.60 | 1124.57 | 1124.02 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 1115.50 | 1122.10 | 1122.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 1108.30 | 1119.34 | 1121.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 11:15:00 | 1075.90 | 1075.24 | 1085.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 11:45:00 | 1076.00 | 1075.24 | 1085.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1086.40 | 1078.58 | 1083.30 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 1106.10 | 1086.66 | 1086.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1136.50 | 1115.92 | 1105.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 10:15:00 | 1238.00 | 1238.71 | 1219.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 10:30:00 | 1234.50 | 1238.71 | 1219.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 1222.00 | 1235.37 | 1220.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 1222.00 | 1235.37 | 1220.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1221.40 | 1232.57 | 1220.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:30:00 | 1226.10 | 1231.54 | 1220.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 1206.70 | 1225.24 | 1220.61 | SL hit (close<static) qty=1.00 sl=1220.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 1257.60 | 1260.88 | 1261.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 1252.00 | 1259.11 | 1260.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 1225.60 | 1224.90 | 1232.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 1225.60 | 1224.90 | 1232.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1174.70 | 1165.48 | 1173.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 1174.70 | 1165.48 | 1173.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1167.10 | 1165.81 | 1172.93 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1189.00 | 1175.61 | 1174.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 1189.20 | 1178.32 | 1176.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 1216.90 | 1219.39 | 1212.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 1216.90 | 1219.39 | 1212.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1207.30 | 1216.28 | 1212.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 1207.30 | 1216.28 | 1212.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1207.90 | 1214.61 | 1211.87 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1201.80 | 1209.05 | 1209.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 1200.20 | 1207.28 | 1208.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 1128.90 | 1126.22 | 1136.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 09:15:00 | 1143.00 | 1126.22 | 1136.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1142.40 | 1129.45 | 1137.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 1143.40 | 1129.45 | 1137.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1143.20 | 1132.20 | 1137.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:15:00 | 1144.30 | 1132.20 | 1137.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1148.00 | 1139.44 | 1139.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 1148.00 | 1139.44 | 1139.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 1151.00 | 1141.75 | 1140.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 1158.40 | 1145.81 | 1142.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 1145.20 | 1147.32 | 1144.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 1145.20 | 1147.32 | 1144.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1146.30 | 1147.11 | 1144.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 1145.10 | 1147.11 | 1144.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1154.00 | 1149.61 | 1146.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 1165.00 | 1152.67 | 1147.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:15:00 | 1162.00 | 1166.30 | 1164.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:30:00 | 1167.00 | 1166.68 | 1165.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 1162.00 | 1165.64 | 1165.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1168.90 | 1166.29 | 1165.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 1164.90 | 1166.29 | 1165.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1165.30 | 1166.09 | 1165.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 1165.30 | 1166.09 | 1165.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1165.20 | 1165.91 | 1165.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:15:00 | 1166.00 | 1165.91 | 1165.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1166.90 | 1166.11 | 1165.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 1167.20 | 1166.11 | 1165.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1168.20 | 1166.53 | 1165.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1102.00 | 1166.53 | 1165.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1112.30 | 1155.68 | 1160.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 1112.30 | 1155.68 | 1160.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 15:15:00 | 1090.00 | 1121.79 | 1139.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 1072.10 | 1070.82 | 1078.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:00:00 | 1072.10 | 1070.82 | 1078.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1070.40 | 1070.45 | 1075.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 1067.80 | 1069.47 | 1074.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 15:15:00 | 1058.40 | 1050.74 | 1050.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 15:15:00 | 1058.40 | 1050.74 | 1050.52 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 1049.80 | 1050.37 | 1050.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 1044.90 | 1049.15 | 1049.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 15:15:00 | 1015.00 | 1012.88 | 1019.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 15:15:00 | 1015.00 | 1012.88 | 1019.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 1015.00 | 1012.88 | 1019.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 1009.50 | 1012.88 | 1019.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 1010.00 | 1012.35 | 1018.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:45:00 | 1010.10 | 1011.86 | 1018.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:45:00 | 1008.90 | 1011.57 | 1017.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1014.10 | 1012.18 | 1015.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:45:00 | 1009.90 | 1012.16 | 1014.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 959.50 | 967.15 | 976.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 959.60 | 967.15 | 976.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 967.25 | 966.64 | 974.59 | SL hit (close>ema200) qty=0.50 sl=966.64 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 982.50 | 962.66 | 961.63 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 962.40 | 969.96 | 969.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 958.00 | 960.45 | 963.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 11:15:00 | 960.35 | 960.13 | 962.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:45:00 | 960.00 | 960.13 | 962.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 941.40 | 939.60 | 943.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 943.00 | 939.60 | 943.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 943.75 | 940.43 | 943.25 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 952.15 | 944.29 | 944.18 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 937.25 | 947.10 | 947.93 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 945.00 | 942.73 | 942.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 947.35 | 943.66 | 943.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 09:15:00 | 944.60 | 946.10 | 944.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 944.60 | 946.10 | 944.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 944.60 | 946.10 | 944.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 944.60 | 946.10 | 944.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 946.05 | 946.09 | 945.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:00:00 | 948.70 | 946.61 | 945.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:45:00 | 948.35 | 946.96 | 945.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:45:00 | 947.95 | 947.72 | 946.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 948.50 | 951.30 | 950.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 946.75 | 949.69 | 949.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 14:15:00 | 946.75 | 949.69 | 949.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 938.60 | 947.15 | 948.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 928.50 | 927.16 | 932.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 10:15:00 | 928.50 | 927.16 | 932.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 928.50 | 927.16 | 932.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 929.70 | 927.16 | 932.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 894.70 | 893.86 | 898.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:45:00 | 898.20 | 893.86 | 898.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 891.15 | 893.32 | 898.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:45:00 | 889.40 | 892.50 | 897.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:30:00 | 887.95 | 891.01 | 896.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 878.70 | 889.88 | 894.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 844.93 | 861.77 | 871.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 843.55 | 861.77 | 871.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 859.60 | 851.69 | 859.88 | SL hit (close>ema200) qty=0.50 sl=851.69 alert=retest2 |

### Cycle 119 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 870.60 | 857.68 | 856.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 873.30 | 864.08 | 860.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 14:15:00 | 866.10 | 866.70 | 863.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 14:45:00 | 865.20 | 866.70 | 863.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 865.00 | 866.36 | 863.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 856.65 | 866.36 | 863.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 856.65 | 864.42 | 862.79 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 850.50 | 859.88 | 860.90 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 872.40 | 862.31 | 861.39 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 860.90 | 863.46 | 863.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 855.10 | 861.23 | 862.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 862.35 | 851.86 | 853.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 862.35 | 851.86 | 853.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 862.35 | 851.86 | 853.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 863.75 | 851.86 | 853.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 867.45 | 854.98 | 855.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 867.45 | 854.98 | 855.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 869.75 | 857.93 | 856.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 874.30 | 861.21 | 857.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 906.15 | 909.01 | 892.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 888.65 | 902.55 | 897.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 888.65 | 902.55 | 897.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 888.65 | 902.55 | 897.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 885.85 | 899.21 | 896.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 885.85 | 899.21 | 896.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 885.85 | 893.95 | 894.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 880.05 | 888.83 | 891.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 839.90 | 839.24 | 854.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 839.90 | 839.24 | 854.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 837.35 | 838.63 | 846.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 833.70 | 838.63 | 846.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 13:45:00 | 834.70 | 836.24 | 842.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:15:00 | 835.40 | 836.24 | 842.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 835.00 | 836.69 | 841.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 806.80 | 799.86 | 804.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 806.40 | 799.86 | 804.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 803.05 | 800.50 | 804.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:15:00 | 796.65 | 800.50 | 804.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:15:00 | 793.63 | 799.70 | 803.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 792.97 | 797.13 | 800.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 793.25 | 797.13 | 800.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 797.60 | 797.22 | 800.52 | SL hit (close>ema200) qty=0.50 sl=797.22 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 659.75 | 646.91 | 645.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 15:15:00 | 664.30 | 654.41 | 649.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 640.15 | 651.56 | 648.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 640.15 | 651.56 | 648.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 640.15 | 651.56 | 648.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 640.15 | 651.56 | 648.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 640.70 | 649.39 | 648.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 640.70 | 649.39 | 648.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 644.00 | 647.13 | 647.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 636.00 | 643.87 | 645.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 636.20 | 621.04 | 629.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 636.20 | 621.04 | 629.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 636.20 | 621.04 | 629.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 636.20 | 621.04 | 629.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 629.70 | 622.77 | 629.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 628.35 | 628.78 | 630.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:00:00 | 628.30 | 628.78 | 630.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 12:15:00 | 640.95 | 630.83 | 630.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 640.95 | 630.83 | 630.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 15:15:00 | 659.00 | 642.85 | 636.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 642.05 | 642.69 | 637.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 10:00:00 | 642.05 | 642.69 | 637.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 723.85 | 714.19 | 703.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 728.00 | 714.19 | 703.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 726.60 | 716.51 | 705.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 728.60 | 718.92 | 707.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 15:15:00 | 756.00 | 761.64 | 761.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 756.00 | 761.64 | 761.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 747.85 | 758.88 | 760.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 728.70 | 728.60 | 734.22 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 15:00:00 | 724.95 | 728.46 | 732.18 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 723.50 | 721.48 | 725.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 723.50 | 721.48 | 725.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 725.60 | 722.31 | 725.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 725.60 | 722.31 | 725.34 | SL hit (close>ema400) qty=1.00 sl=725.34 alert=retest1 |

### Cycle 129 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 724.50 | 722.71 | 722.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 731.95 | 724.56 | 723.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 12:15:00 | 724.55 | 726.09 | 724.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 12:15:00 | 724.55 | 726.09 | 724.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 724.55 | 726.09 | 724.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:45:00 | 723.85 | 726.09 | 724.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 729.05 | 726.69 | 725.03 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 721.95 | 724.24 | 724.43 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 1343.90 | 2024-05-23 13:15:00 | 1355.30 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2024-05-31 09:15:00 | 1382.35 | 2024-05-31 14:15:00 | 1368.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest1 | 2024-06-11 09:15:00 | 1489.35 | 2024-06-12 10:15:00 | 1474.05 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest1 | 2024-06-11 13:30:00 | 1486.40 | 2024-06-12 10:15:00 | 1474.05 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1476.35 | 2024-06-12 12:15:00 | 1467.35 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-06-12 11:30:00 | 1477.35 | 2024-06-12 12:15:00 | 1467.35 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-06-19 10:15:00 | 1448.45 | 2024-06-20 11:15:00 | 1464.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-06-19 15:00:00 | 1448.65 | 2024-06-20 11:15:00 | 1464.60 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-07-04 09:45:00 | 1497.00 | 2024-07-12 15:15:00 | 1514.00 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2024-07-04 11:15:00 | 1496.30 | 2024-07-12 15:15:00 | 1514.00 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2024-07-04 11:45:00 | 1495.00 | 2024-07-12 15:15:00 | 1514.00 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2024-07-04 13:15:00 | 1497.60 | 2024-07-12 15:15:00 | 1514.00 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2024-07-10 12:30:00 | 1526.30 | 2024-07-12 15:15:00 | 1514.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-07-12 11:45:00 | 1527.00 | 2024-07-12 15:15:00 | 1514.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-07-30 11:15:00 | 1602.75 | 2024-07-31 11:15:00 | 1600.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-07-31 10:45:00 | 1604.55 | 2024-07-31 11:15:00 | 1600.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-08-21 09:15:00 | 1423.85 | 2024-08-21 09:15:00 | 1417.45 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-08-27 13:30:00 | 1451.45 | 2024-08-28 15:15:00 | 1444.05 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-08-27 14:30:00 | 1450.05 | 2024-08-28 15:15:00 | 1444.05 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-08-29 09:15:00 | 1450.15 | 2024-08-29 12:15:00 | 1439.05 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-08-29 11:00:00 | 1452.75 | 2024-08-29 12:15:00 | 1439.05 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-09-12 13:45:00 | 1421.45 | 2024-09-13 09:15:00 | 1434.25 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-09-18 09:15:00 | 1443.65 | 2024-09-18 11:15:00 | 1430.80 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-09-24 13:15:00 | 1432.80 | 2024-09-25 09:15:00 | 1423.40 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-09-26 09:15:00 | 1419.10 | 2024-09-27 09:15:00 | 1446.30 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-09-26 11:00:00 | 1419.10 | 2024-09-27 09:15:00 | 1446.30 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-09-26 11:45:00 | 1420.45 | 2024-09-27 09:15:00 | 1446.30 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-09-26 13:00:00 | 1418.85 | 2024-09-27 09:15:00 | 1446.30 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-10-01 09:15:00 | 1441.90 | 2024-10-01 11:15:00 | 1435.20 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-10-11 10:15:00 | 1400.00 | 2024-10-21 13:15:00 | 1429.00 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest2 | 2024-10-11 11:00:00 | 1402.30 | 2024-10-21 13:15:00 | 1429.00 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2024-10-14 10:30:00 | 1401.65 | 2024-10-21 13:15:00 | 1429.00 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest2 | 2024-10-25 10:30:00 | 1363.35 | 2024-11-05 09:15:00 | 1301.55 | PARTIAL | 0.50 | 4.53% |
| SELL | retest2 | 2024-10-25 13:30:00 | 1364.00 | 2024-11-05 09:15:00 | 1302.73 | PARTIAL | 0.50 | 4.49% |
| SELL | retest2 | 2024-10-25 14:00:00 | 1363.00 | 2024-11-05 09:15:00 | 1303.92 | PARTIAL | 0.50 | 4.33% |
| SELL | retest2 | 2024-10-25 15:00:00 | 1363.75 | 2024-11-05 09:15:00 | 1301.88 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2024-10-25 10:30:00 | 1363.35 | 2024-11-05 10:15:00 | 1340.00 | STOP_HIT | 0.50 | 1.71% |
| SELL | retest2 | 2024-10-25 13:30:00 | 1364.00 | 2024-11-05 10:15:00 | 1340.00 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2024-10-25 14:00:00 | 1363.00 | 2024-11-05 10:15:00 | 1340.00 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2024-10-25 15:00:00 | 1363.75 | 2024-11-05 10:15:00 | 1340.00 | STOP_HIT | 0.50 | 1.74% |
| SELL | retest2 | 2024-10-28 15:15:00 | 1370.05 | 2024-11-06 10:15:00 | 1355.00 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2024-10-30 10:45:00 | 1371.30 | 2024-11-06 10:15:00 | 1355.00 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2024-10-30 11:15:00 | 1372.55 | 2024-11-06 10:15:00 | 1355.00 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2024-10-30 13:45:00 | 1370.40 | 2024-11-06 10:15:00 | 1355.00 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2024-11-04 09:15:00 | 1345.60 | 2024-11-06 10:15:00 | 1355.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-11-04 09:45:00 | 1350.65 | 2024-11-06 10:15:00 | 1355.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-11-14 13:30:00 | 1325.85 | 2024-11-22 15:15:00 | 1299.30 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2024-11-29 09:15:00 | 1377.00 | 2024-12-10 13:15:00 | 1432.00 | STOP_HIT | 1.00 | 3.99% |
| BUY | retest2 | 2024-11-29 09:45:00 | 1379.80 | 2024-12-10 13:15:00 | 1432.00 | STOP_HIT | 1.00 | 3.78% |
| SELL | retest1 | 2024-12-13 09:15:00 | 1424.05 | 2024-12-13 14:15:00 | 1432.75 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-12-16 12:15:00 | 1427.20 | 2024-12-16 13:15:00 | 1435.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-12-26 09:15:00 | 1353.75 | 2024-12-27 09:15:00 | 1376.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-12-26 13:15:00 | 1359.00 | 2024-12-27 09:15:00 | 1376.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-12-30 15:00:00 | 1389.15 | 2025-01-08 09:15:00 | 1408.45 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-01-01 10:00:00 | 1391.60 | 2025-01-08 09:15:00 | 1408.45 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest2 | 2025-01-21 10:15:00 | 1293.10 | 2025-01-23 14:15:00 | 1299.60 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-01-21 12:00:00 | 1298.10 | 2025-01-23 14:15:00 | 1299.60 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-01-23 14:15:00 | 1298.05 | 2025-01-23 14:15:00 | 1299.60 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-01-31 09:15:00 | 1290.40 | 2025-02-03 13:15:00 | 1419.44 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-11 13:00:00 | 1327.70 | 2025-02-12 11:15:00 | 1377.55 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest1 | 2025-02-18 09:15:00 | 1282.05 | 2025-02-21 09:15:00 | 1289.45 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-02-18 14:15:00 | 1283.00 | 2025-02-21 09:15:00 | 1289.45 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-02-19 10:45:00 | 1262.55 | 2025-02-21 10:15:00 | 1307.95 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-03-04 09:15:00 | 1212.40 | 2025-03-06 09:15:00 | 1250.35 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-03-05 11:00:00 | 1230.65 | 2025-03-06 09:15:00 | 1250.35 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-03-05 12:00:00 | 1229.85 | 2025-03-06 09:15:00 | 1250.35 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-03-05 13:30:00 | 1230.55 | 2025-03-06 09:15:00 | 1250.35 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-03-12 11:15:00 | 1208.45 | 2025-03-12 14:15:00 | 1229.85 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-03-12 12:45:00 | 1211.45 | 2025-03-12 14:15:00 | 1229.85 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-03-26 11:00:00 | 1227.05 | 2025-04-01 09:15:00 | 1237.80 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-03-26 11:30:00 | 1226.10 | 2025-04-01 09:15:00 | 1237.80 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest1 | 2025-04-07 09:15:00 | 1178.00 | 2025-04-08 09:15:00 | 1210.45 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-04-16 09:15:00 | 1234.00 | 2025-04-17 12:15:00 | 1223.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-04-16 11:00:00 | 1233.60 | 2025-04-17 12:15:00 | 1223.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-04-16 12:00:00 | 1232.70 | 2025-04-17 12:15:00 | 1223.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-04-16 12:30:00 | 1234.00 | 2025-04-17 12:15:00 | 1223.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-04-23 10:00:00 | 1218.10 | 2025-04-23 10:15:00 | 1229.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-04-24 14:15:00 | 1228.20 | 2025-04-25 09:15:00 | 1212.90 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-05-02 12:15:00 | 1194.70 | 2025-05-06 10:15:00 | 1213.50 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-05-02 12:45:00 | 1193.70 | 2025-05-06 10:15:00 | 1213.50 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-05-02 13:30:00 | 1194.30 | 2025-05-06 10:15:00 | 1213.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-05-12 10:45:00 | 1209.10 | 2025-05-12 13:15:00 | 1211.80 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-05-12 12:45:00 | 1208.40 | 2025-05-12 13:15:00 | 1211.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-05-27 11:30:00 | 1285.00 | 2025-05-28 09:15:00 | 1263.50 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1284.10 | 2025-05-29 10:15:00 | 1265.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-06-02 12:30:00 | 1254.80 | 2025-06-12 10:15:00 | 1223.90 | STOP_HIT | 1.00 | 2.46% |
| SELL | retest2 | 2025-06-03 09:15:00 | 1255.90 | 2025-06-12 10:15:00 | 1223.90 | STOP_HIT | 1.00 | 2.55% |
| SELL | retest2 | 2025-06-23 13:00:00 | 1197.80 | 2025-06-25 09:15:00 | 1211.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-23 15:00:00 | 1196.30 | 2025-06-25 09:15:00 | 1211.20 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-26 14:15:00 | 1223.60 | 2025-07-07 15:15:00 | 1225.00 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-07-25 10:00:00 | 1205.60 | 2025-08-11 11:15:00 | 1186.50 | STOP_HIT | 1.00 | 1.58% |
| SELL | retest2 | 2025-07-25 15:15:00 | 1203.90 | 2025-08-11 11:15:00 | 1186.50 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2025-09-08 13:30:00 | 1226.10 | 2025-09-09 09:15:00 | 1206.70 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1233.60 | 2025-09-18 10:15:00 | 1257.60 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2025-09-09 11:15:00 | 1226.60 | 2025-09-18 10:15:00 | 1257.60 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2025-09-09 12:30:00 | 1225.60 | 2025-09-18 10:15:00 | 1257.60 | STOP_HIT | 1.00 | 2.61% |
| BUY | retest2 | 2025-09-11 11:45:00 | 1245.50 | 2025-09-18 10:15:00 | 1257.60 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-09-12 09:45:00 | 1247.00 | 2025-09-18 10:15:00 | 1257.60 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2025-10-20 10:45:00 | 1165.00 | 2025-10-28 09:15:00 | 1112.30 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest2 | 2025-10-24 14:15:00 | 1162.00 | 2025-10-28 09:15:00 | 1112.30 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-10-27 09:30:00 | 1167.00 | 2025-10-28 09:15:00 | 1112.30 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest2 | 2025-10-27 10:45:00 | 1162.00 | 2025-10-28 09:15:00 | 1112.30 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-11-04 11:30:00 | 1067.80 | 2025-11-12 15:15:00 | 1058.40 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-11-19 09:15:00 | 1009.50 | 2025-12-03 09:15:00 | 959.50 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-11-19 10:15:00 | 1010.00 | 2025-12-03 09:15:00 | 959.60 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-11-19 09:15:00 | 1009.50 | 2025-12-03 11:15:00 | 967.25 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2025-11-19 10:15:00 | 1010.00 | 2025-12-03 11:15:00 | 967.25 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2025-11-19 10:45:00 | 1010.10 | 2025-12-04 14:15:00 | 959.02 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-11-19 11:45:00 | 1008.90 | 2025-12-04 14:15:00 | 958.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 11:45:00 | 1009.90 | 2025-12-04 14:15:00 | 959.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 10:45:00 | 1010.10 | 2025-12-05 11:15:00 | 965.00 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-11-19 11:45:00 | 1008.90 | 2025-12-05 11:15:00 | 965.00 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2025-11-20 11:45:00 | 1009.90 | 2025-12-05 11:15:00 | 965.00 | STOP_HIT | 0.50 | 4.45% |
| BUY | retest2 | 2026-01-02 12:00:00 | 948.70 | 2026-01-06 14:15:00 | 946.75 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2026-01-02 12:45:00 | 948.35 | 2026-01-06 14:15:00 | 946.75 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-01-02 14:45:00 | 947.95 | 2026-01-06 14:15:00 | 946.75 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-01-06 13:00:00 | 948.50 | 2026-01-06 14:15:00 | 946.75 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-01-20 12:45:00 | 889.40 | 2026-01-27 09:15:00 | 844.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 13:30:00 | 887.95 | 2026-01-27 09:15:00 | 843.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:45:00 | 889.40 | 2026-01-28 09:15:00 | 859.60 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2026-01-20 13:30:00 | 887.95 | 2026-01-28 09:15:00 | 859.60 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-01-21 09:15:00 | 878.70 | 2026-01-30 10:15:00 | 870.60 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2026-02-18 10:15:00 | 833.70 | 2026-02-26 11:15:00 | 793.63 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2026-02-18 13:45:00 | 834.70 | 2026-02-26 15:15:00 | 792.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 14:15:00 | 835.40 | 2026-02-26 15:15:00 | 793.25 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2026-02-18 10:15:00 | 833.70 | 2026-02-27 09:15:00 | 797.60 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2026-02-18 13:45:00 | 834.70 | 2026-02-27 09:15:00 | 797.60 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2026-02-18 14:15:00 | 835.40 | 2026-02-27 09:15:00 | 797.60 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2026-02-19 09:15:00 | 835.00 | 2026-02-27 10:15:00 | 792.01 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2026-02-26 11:15:00 | 796.65 | 2026-03-04 09:15:00 | 756.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:15:00 | 835.00 | 2026-03-04 10:15:00 | 750.33 | TARGET_HIT | 0.50 | 10.14% |
| SELL | retest2 | 2026-02-26 11:15:00 | 796.65 | 2026-03-05 11:15:00 | 743.15 | STOP_HIT | 0.50 | 6.72% |
| SELL | retest2 | 2026-04-01 14:30:00 | 628.35 | 2026-04-02 12:15:00 | 640.95 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-04-01 15:00:00 | 628.30 | 2026-04-02 12:15:00 | 640.95 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-04-13 10:15:00 | 728.00 | 2026-04-23 15:15:00 | 756.00 | STOP_HIT | 1.00 | 3.85% |
| BUY | retest2 | 2026-04-13 10:45:00 | 726.60 | 2026-04-23 15:15:00 | 756.00 | STOP_HIT | 1.00 | 4.05% |
| BUY | retest2 | 2026-04-13 12:00:00 | 728.60 | 2026-04-23 15:15:00 | 756.00 | STOP_HIT | 1.00 | 3.76% |
| SELL | retest1 | 2026-04-29 15:00:00 | 724.95 | 2026-05-04 10:15:00 | 725.60 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2026-05-04 12:45:00 | 723.00 | 2026-05-06 15:15:00 | 724.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2026-05-05 09:15:00 | 722.90 | 2026-05-06 15:15:00 | 724.50 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-05-05 10:00:00 | 723.35 | 2026-05-06 15:15:00 | 724.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2026-05-05 13:15:00 | 717.35 | 2026-05-06 15:15:00 | 724.50 | STOP_HIT | 1.00 | -1.00% |
