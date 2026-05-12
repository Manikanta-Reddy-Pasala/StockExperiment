# Mahanagar Gas Ltd. (MGL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1173.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 159 |
| ALERT1 | 109 |
| ALERT2 | 107 |
| ALERT2_SKIP | 54 |
| ALERT3 | 257 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 111 |
| PARTIAL | 5 |
| TARGET_HIT | 9 |
| STOP_HIT | 105 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 117 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 34 / 83
- **Target hits / Stop hits / Partials:** 9 / 103 / 5
- **Avg / median % per leg:** 0.08% / -0.89%
- **Sum % (uncompounded):** 9.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 16 | 31.4% | 8 | 43 | 0 | 0.92% | 47.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.04% | -2.1% |
| BUY @ 3rd Alert (retest2) | 49 | 16 | 32.7% | 8 | 41 | 0 | 1.00% | 49.1% |
| SELL (all) | 66 | 18 | 27.3% | 1 | 60 | 5 | -0.56% | -37.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 66 | 18 | 27.3% | 1 | 60 | 5 | -0.56% | -37.2% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.04% | -2.1% |
| retest2 (combined) | 115 | 34 | 29.6% | 9 | 101 | 5 | 0.10% | 11.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 1312.90 | 1304.21 | 1303.71 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 11:15:00 | 1298.00 | 1302.71 | 1303.11 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 14:15:00 | 1318.00 | 1306.23 | 1304.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 15:15:00 | 1323.00 | 1309.58 | 1306.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1304.95 | 1312.88 | 1309.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 1304.95 | 1312.88 | 1309.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1304.95 | 1312.88 | 1309.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 1306.30 | 1312.88 | 1309.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1304.75 | 1311.25 | 1308.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 1306.00 | 1311.25 | 1308.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 1309.90 | 1310.98 | 1309.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:15:00 | 1310.00 | 1310.98 | 1309.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 13:15:00 | 1301.45 | 1307.82 | 1307.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 1301.45 | 1307.82 | 1307.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 14:15:00 | 1297.70 | 1305.80 | 1306.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 15:15:00 | 1300.00 | 1299.98 | 1302.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 09:15:00 | 1305.90 | 1299.98 | 1302.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1296.05 | 1299.19 | 1302.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:30:00 | 1302.05 | 1299.19 | 1302.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1290.70 | 1293.95 | 1297.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 1295.50 | 1293.95 | 1297.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1300.90 | 1295.34 | 1297.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 1300.90 | 1295.34 | 1297.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 1296.00 | 1295.47 | 1297.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:00:00 | 1293.25 | 1295.03 | 1297.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 14:15:00 | 1291.80 | 1295.41 | 1297.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:45:00 | 1291.90 | 1294.48 | 1296.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 11:15:00 | 1302.45 | 1296.69 | 1296.91 | SL hit (close>static) qty=1.00 sl=1302.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 12:15:00 | 1303.90 | 1298.13 | 1297.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 13:15:00 | 1311.40 | 1300.78 | 1298.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 15:15:00 | 1300.80 | 1301.39 | 1299.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 09:15:00 | 1300.00 | 1301.39 | 1299.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1292.90 | 1299.69 | 1298.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 1292.90 | 1299.69 | 1298.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1293.00 | 1298.35 | 1298.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:00:00 | 1293.00 | 1298.35 | 1298.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 1290.50 | 1296.78 | 1297.62 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 14:15:00 | 1304.25 | 1297.80 | 1297.03 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 1290.90 | 1295.97 | 1296.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 1281.50 | 1292.43 | 1294.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 1293.75 | 1290.28 | 1292.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 1293.75 | 1290.28 | 1292.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1293.75 | 1290.28 | 1292.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 1291.15 | 1290.28 | 1292.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1311.00 | 1294.42 | 1294.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 1311.00 | 1294.42 | 1294.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 1290.00 | 1293.54 | 1294.14 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 1298.25 | 1294.92 | 1294.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1374.60 | 1311.51 | 1302.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1324.60 | 1351.20 | 1332.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1324.60 | 1351.20 | 1332.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1324.60 | 1351.20 | 1332.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1314.90 | 1351.20 | 1332.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1271.80 | 1335.32 | 1327.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1271.80 | 1335.32 | 1327.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1240.75 | 1316.41 | 1319.19 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1368.75 | 1316.42 | 1310.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1382.55 | 1356.01 | 1336.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 1401.05 | 1404.25 | 1386.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 1395.90 | 1404.25 | 1386.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1450.45 | 1467.51 | 1461.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 1454.00 | 1467.51 | 1461.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1452.65 | 1464.54 | 1460.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 1450.65 | 1464.54 | 1460.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 1446.70 | 1458.50 | 1458.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 1445.00 | 1455.80 | 1457.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 1473.60 | 1457.86 | 1458.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 1473.60 | 1457.86 | 1458.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1473.60 | 1457.86 | 1458.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 1473.60 | 1457.86 | 1458.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 1470.20 | 1460.33 | 1459.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 11:15:00 | 1476.00 | 1463.46 | 1460.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 1490.90 | 1495.97 | 1482.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 12:45:00 | 1493.35 | 1495.97 | 1482.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1475.75 | 1491.57 | 1483.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 1476.50 | 1491.57 | 1483.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1487.50 | 1490.75 | 1484.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 1480.20 | 1490.75 | 1484.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 1491.20 | 1495.34 | 1488.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:45:00 | 1492.65 | 1495.34 | 1488.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 1487.05 | 1493.68 | 1488.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 1487.05 | 1493.68 | 1488.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 1487.00 | 1492.34 | 1488.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 1501.00 | 1492.34 | 1488.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1500.50 | 1493.97 | 1489.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:30:00 | 1517.40 | 1500.23 | 1496.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 1549.65 | 1501.10 | 1498.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-01 11:15:00 | 1669.14 | 1615.77 | 1573.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 1664.60 | 1680.59 | 1681.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 1662.20 | 1674.75 | 1678.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 12:15:00 | 1686.40 | 1673.99 | 1677.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 12:15:00 | 1686.40 | 1673.99 | 1677.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1686.40 | 1673.99 | 1677.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:00:00 | 1686.40 | 1673.99 | 1677.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 1691.65 | 1677.52 | 1678.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:00:00 | 1691.65 | 1677.52 | 1678.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 1700.65 | 1682.15 | 1680.45 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 1657.65 | 1675.75 | 1678.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 12:15:00 | 1647.00 | 1670.00 | 1675.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 1675.95 | 1668.33 | 1672.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 1675.95 | 1668.33 | 1672.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1675.95 | 1668.33 | 1672.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 10:30:00 | 1658.60 | 1668.21 | 1671.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 11:45:00 | 1657.60 | 1666.98 | 1671.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 1682.15 | 1672.59 | 1671.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 10:15:00 | 1682.15 | 1672.59 | 1671.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 13:15:00 | 1715.70 | 1690.06 | 1680.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 09:15:00 | 1695.00 | 1698.45 | 1687.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 1695.00 | 1698.45 | 1687.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1695.00 | 1698.45 | 1687.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 10:30:00 | 1723.65 | 1701.88 | 1694.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 10:00:00 | 1738.95 | 1708.18 | 1700.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 14:15:00 | 1719.75 | 1728.93 | 1728.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 1703.85 | 1723.67 | 1725.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 1703.85 | 1723.67 | 1725.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 1691.90 | 1709.19 | 1717.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1737.65 | 1712.93 | 1717.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1737.65 | 1712.93 | 1717.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1737.65 | 1712.93 | 1717.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 1736.50 | 1712.93 | 1717.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1735.45 | 1717.43 | 1719.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 1739.25 | 1717.43 | 1719.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 1741.95 | 1722.34 | 1721.09 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 1719.95 | 1723.25 | 1723.60 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 1750.00 | 1727.28 | 1725.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 12:15:00 | 1761.60 | 1748.39 | 1740.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 1748.05 | 1750.78 | 1743.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 15:00:00 | 1748.05 | 1750.78 | 1743.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 1794.05 | 1761.23 | 1749.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:45:00 | 1815.40 | 1775.58 | 1758.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 12:15:00 | 1822.85 | 1775.58 | 1758.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 12:15:00 | 1831.70 | 1855.60 | 1856.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 1831.70 | 1855.60 | 1856.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 1824.40 | 1845.05 | 1851.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 1839.15 | 1838.82 | 1846.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 10:45:00 | 1841.90 | 1838.82 | 1846.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 1839.00 | 1838.80 | 1845.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 12:30:00 | 1841.80 | 1838.80 | 1845.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 1855.90 | 1842.22 | 1845.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:00:00 | 1855.90 | 1842.22 | 1845.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 1846.90 | 1843.16 | 1846.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 1822.10 | 1844.13 | 1846.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 10:15:00 | 1828.25 | 1845.15 | 1846.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:00:00 | 1818.00 | 1806.08 | 1820.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 1814.05 | 1808.31 | 1808.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 1814.05 | 1808.31 | 1808.15 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 1801.70 | 1808.37 | 1808.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 15:15:00 | 1798.10 | 1806.31 | 1807.54 | Break + close below crossover candle low |

### Cycle 25 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 1821.55 | 1809.36 | 1808.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 15:15:00 | 1833.00 | 1820.77 | 1816.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 11:15:00 | 1820.20 | 1822.64 | 1818.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 12:00:00 | 1820.20 | 1822.64 | 1818.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1809.10 | 1819.94 | 1817.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 1811.70 | 1819.94 | 1817.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 1794.75 | 1814.90 | 1815.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 1770.80 | 1798.84 | 1807.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1770.70 | 1760.42 | 1779.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1770.70 | 1760.42 | 1779.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1770.70 | 1760.42 | 1779.54 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 1817.45 | 1787.74 | 1786.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 15:15:00 | 1820.00 | 1794.19 | 1789.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 1830.95 | 1836.53 | 1821.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:00:00 | 1830.95 | 1836.53 | 1821.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1824.85 | 1834.20 | 1822.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 1822.10 | 1834.20 | 1822.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1853.60 | 1838.89 | 1828.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 13:30:00 | 1861.15 | 1849.38 | 1837.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 15:00:00 | 1857.95 | 1851.09 | 1839.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 09:15:00 | 1827.00 | 1846.73 | 1839.44 | SL hit (close<static) qty=1.00 sl=1827.60 alert=retest2 |

### Cycle 28 — SELL (started 2024-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 12:15:00 | 1816.25 | 1835.53 | 1835.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 09:15:00 | 1792.00 | 1818.49 | 1826.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 14:15:00 | 1773.95 | 1771.50 | 1786.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 14:30:00 | 1773.15 | 1771.50 | 1786.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1789.75 | 1776.67 | 1783.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:00:00 | 1789.75 | 1776.67 | 1783.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 1780.85 | 1777.51 | 1783.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 15:15:00 | 1773.00 | 1778.01 | 1782.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1769.90 | 1777.98 | 1782.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 10:15:00 | 1792.10 | 1780.81 | 1782.96 | SL hit (close>static) qty=1.00 sl=1790.95 alert=retest2 |

### Cycle 29 — BUY (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 13:15:00 | 1791.80 | 1785.76 | 1784.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 09:15:00 | 1806.25 | 1790.37 | 1787.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 1770.10 | 1786.32 | 1785.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 10:15:00 | 1770.10 | 1786.32 | 1785.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1770.10 | 1786.32 | 1785.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:00:00 | 1770.10 | 1786.32 | 1785.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 1772.60 | 1783.57 | 1784.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 12:15:00 | 1763.35 | 1779.53 | 1782.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 1778.00 | 1776.20 | 1780.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 15:15:00 | 1778.00 | 1776.20 | 1780.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1778.00 | 1776.20 | 1780.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 1811.40 | 1776.20 | 1780.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1820.50 | 1785.06 | 1783.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 1864.05 | 1831.08 | 1816.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 1853.80 | 1868.94 | 1854.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 1853.80 | 1868.94 | 1854.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1853.80 | 1868.94 | 1854.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 1853.80 | 1868.94 | 1854.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 1857.90 | 1866.73 | 1854.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:15:00 | 1872.15 | 1866.73 | 1854.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 13:15:00 | 1841.85 | 1860.19 | 1854.55 | SL hit (close<static) qty=1.00 sl=1843.20 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 1833.05 | 1851.17 | 1851.20 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 09:15:00 | 1867.20 | 1854.38 | 1852.66 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 1831.00 | 1850.23 | 1851.37 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 1890.00 | 1854.80 | 1852.61 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 10:15:00 | 1839.05 | 1856.74 | 1857.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 11:15:00 | 1829.35 | 1851.26 | 1854.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 1829.65 | 1827.14 | 1839.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 1829.65 | 1827.14 | 1839.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1829.65 | 1827.14 | 1839.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:30:00 | 1822.45 | 1827.20 | 1835.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 1817.30 | 1827.73 | 1834.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 11:30:00 | 1819.25 | 1826.11 | 1832.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 1823.30 | 1814.25 | 1820.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 1826.45 | 1816.69 | 1820.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 1826.45 | 1816.69 | 1820.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 1821.15 | 1817.58 | 1820.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 1911.00 | 1836.68 | 1828.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 1911.00 | 1836.68 | 1828.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 1931.00 | 1903.63 | 1893.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 15:15:00 | 1922.00 | 1925.88 | 1911.34 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:00:00 | 1945.00 | 1929.70 | 1914.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 11:00:00 | 1939.75 | 1931.71 | 1916.71 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 1922.10 | 1932.94 | 1923.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-24 15:15:00 | 1922.10 | 1932.94 | 1923.74 | SL hit (close<ema400) qty=1.00 sl=1923.74 alert=retest1 |

### Cycle 38 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 1943.55 | 1948.27 | 1948.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1935.65 | 1945.85 | 1947.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 10:15:00 | 1945.85 | 1928.21 | 1934.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 10:15:00 | 1945.85 | 1928.21 | 1934.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 1945.85 | 1928.21 | 1934.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 1945.85 | 1928.21 | 1934.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 1946.75 | 1931.92 | 1935.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:45:00 | 1945.55 | 1931.92 | 1935.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 14:15:00 | 1948.60 | 1938.37 | 1937.72 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 1908.30 | 1931.93 | 1934.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 15:15:00 | 1895.05 | 1914.76 | 1924.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 1849.95 | 1848.60 | 1874.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 1849.95 | 1848.60 | 1874.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1839.90 | 1840.21 | 1858.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:00:00 | 1827.50 | 1837.00 | 1853.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:15:00 | 1828.00 | 1835.75 | 1850.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 1878.10 | 1839.77 | 1848.07 | SL hit (close>static) qty=1.00 sl=1860.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 14:15:00 | 1866.30 | 1855.35 | 1854.13 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 1834.25 | 1851.42 | 1852.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 1815.90 | 1835.70 | 1843.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 1841.75 | 1812.85 | 1824.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1841.75 | 1812.85 | 1824.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1841.75 | 1812.85 | 1824.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 1841.75 | 1812.85 | 1824.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1843.45 | 1818.97 | 1826.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 11:30:00 | 1830.00 | 1821.20 | 1826.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-18 09:15:00 | 1647.00 | 1721.68 | 1763.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 1431.60 | 1404.27 | 1402.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 1434.65 | 1410.35 | 1405.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 10:15:00 | 1433.10 | 1434.72 | 1425.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 11:00:00 | 1433.10 | 1434.72 | 1425.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 1423.60 | 1432.50 | 1425.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 1423.60 | 1432.50 | 1425.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 1426.55 | 1431.31 | 1425.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:15:00 | 1422.05 | 1431.31 | 1425.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 1414.70 | 1427.99 | 1424.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:30:00 | 1414.50 | 1427.99 | 1424.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 1422.00 | 1426.79 | 1424.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 1424.25 | 1426.79 | 1424.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 1416.05 | 1424.64 | 1423.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 1409.35 | 1424.64 | 1423.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1406.35 | 1420.98 | 1422.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 1382.95 | 1402.98 | 1411.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 14:15:00 | 1122.80 | 1122.00 | 1155.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-21 15:00:00 | 1122.80 | 1122.00 | 1155.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1128.15 | 1124.94 | 1150.99 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1190.00 | 1158.32 | 1158.03 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 1169.35 | 1172.49 | 1172.51 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 15:15:00 | 1178.95 | 1173.54 | 1172.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 1190.00 | 1176.84 | 1174.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 1174.65 | 1176.40 | 1174.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 1174.65 | 1176.40 | 1174.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1174.65 | 1176.40 | 1174.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 1175.55 | 1176.40 | 1174.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1177.90 | 1176.70 | 1174.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 13:45:00 | 1179.90 | 1175.81 | 1174.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 15:00:00 | 1179.65 | 1176.57 | 1175.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:30:00 | 1179.90 | 1178.02 | 1176.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:15:00 | 1179.75 | 1177.76 | 1176.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 1175.65 | 1177.33 | 1176.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:00:00 | 1175.65 | 1177.33 | 1176.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 1188.55 | 1179.58 | 1177.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 15:15:00 | 1201.45 | 1181.59 | 1178.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-05 13:15:00 | 1297.89 | 1279.43 | 1263.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 1275.95 | 1284.96 | 1285.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1260.65 | 1277.49 | 1281.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 1285.40 | 1277.54 | 1281.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 1285.40 | 1277.54 | 1281.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 1285.40 | 1277.54 | 1281.03 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 1300.70 | 1286.05 | 1284.35 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 1268.75 | 1283.84 | 1284.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 1256.65 | 1275.59 | 1280.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 09:15:00 | 1247.85 | 1242.01 | 1252.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 10:00:00 | 1247.85 | 1242.01 | 1252.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 1261.00 | 1245.81 | 1253.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 10:30:00 | 1260.95 | 1245.81 | 1253.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 1262.05 | 1249.06 | 1254.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 1261.00 | 1249.06 | 1254.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 1256.65 | 1256.23 | 1256.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 1247.65 | 1256.23 | 1256.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 11:15:00 | 1256.50 | 1254.22 | 1255.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:00:00 | 1251.55 | 1253.69 | 1255.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 10:30:00 | 1255.25 | 1237.39 | 1238.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 11:15:00 | 1259.85 | 1241.89 | 1240.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 1259.85 | 1241.89 | 1240.52 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 11:15:00 | 1230.65 | 1241.50 | 1241.80 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 1252.75 | 1240.36 | 1239.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 1263.00 | 1244.89 | 1242.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 1274.45 | 1276.74 | 1266.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 09:45:00 | 1275.30 | 1276.74 | 1266.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1289.50 | 1307.46 | 1299.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 1289.50 | 1307.46 | 1299.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1288.90 | 1303.74 | 1298.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:30:00 | 1280.45 | 1303.74 | 1298.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 1270.85 | 1292.32 | 1294.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 09:15:00 | 1263.70 | 1281.43 | 1288.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 1264.60 | 1255.11 | 1268.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 10:00:00 | 1264.60 | 1255.11 | 1268.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1255.00 | 1255.09 | 1267.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:30:00 | 1247.10 | 1252.69 | 1263.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:00:00 | 1245.20 | 1252.69 | 1263.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 15:15:00 | 1248.00 | 1253.39 | 1262.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 1288.50 | 1259.55 | 1263.57 | SL hit (close>static) qty=1.00 sl=1272.90 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 12:15:00 | 1281.70 | 1268.35 | 1267.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 13:15:00 | 1298.50 | 1274.38 | 1269.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 14:15:00 | 1273.60 | 1274.22 | 1270.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 14:15:00 | 1273.60 | 1274.22 | 1270.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 1273.60 | 1274.22 | 1270.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 15:00:00 | 1273.60 | 1274.22 | 1270.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 1262.00 | 1271.78 | 1269.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 1256.00 | 1271.78 | 1269.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1271.00 | 1271.62 | 1269.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:45:00 | 1309.80 | 1278.20 | 1273.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-13 10:15:00 | 1249.75 | 1273.25 | 1274.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 1249.75 | 1273.25 | 1274.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 1238.10 | 1262.49 | 1268.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 1266.75 | 1250.07 | 1259.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 1266.75 | 1250.07 | 1259.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1266.75 | 1250.07 | 1259.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 1277.25 | 1250.07 | 1259.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 13:15:00 | 1275.45 | 1267.10 | 1266.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 09:15:00 | 1279.95 | 1270.19 | 1267.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 11:15:00 | 1286.85 | 1292.38 | 1284.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 12:00:00 | 1286.85 | 1292.38 | 1284.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 1323.85 | 1299.73 | 1290.78 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 1297.50 | 1303.67 | 1303.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 12:15:00 | 1287.50 | 1300.44 | 1302.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1282.25 | 1275.39 | 1284.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 1282.25 | 1275.39 | 1284.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 1284.70 | 1277.25 | 1284.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 1270.80 | 1277.25 | 1284.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1282.65 | 1278.33 | 1284.67 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 1304.40 | 1289.01 | 1288.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 14:15:00 | 1312.55 | 1293.71 | 1290.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 1286.20 | 1295.62 | 1292.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 1286.20 | 1295.62 | 1292.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1286.20 | 1295.62 | 1292.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 1286.20 | 1295.62 | 1292.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1296.50 | 1295.79 | 1292.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:30:00 | 1302.60 | 1297.09 | 1293.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 13:15:00 | 1279.95 | 1291.80 | 1291.45 | SL hit (close<static) qty=1.00 sl=1283.90 alert=retest2 |

### Cycle 60 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 1275.75 | 1288.59 | 1290.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1267.45 | 1282.19 | 1286.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 1273.10 | 1264.12 | 1270.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 12:15:00 | 1273.10 | 1264.12 | 1270.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1273.10 | 1264.12 | 1270.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 1273.10 | 1264.12 | 1270.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1264.50 | 1264.20 | 1270.33 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 1301.05 | 1275.14 | 1274.13 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 15:15:00 | 1259.00 | 1272.21 | 1273.54 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 1317.85 | 1281.33 | 1277.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 1344.00 | 1308.70 | 1295.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 1351.55 | 1357.01 | 1335.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 1351.55 | 1357.01 | 1335.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1351.55 | 1357.01 | 1335.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1351.70 | 1357.01 | 1335.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1336.20 | 1352.85 | 1335.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1336.20 | 1352.85 | 1335.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1364.00 | 1355.08 | 1338.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1367.70 | 1355.08 | 1338.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 1313.90 | 1351.79 | 1341.49 | SL hit (close<static) qty=1.00 sl=1331.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 1314.90 | 1332.70 | 1334.74 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 1370.05 | 1339.22 | 1335.38 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 1328.15 | 1339.80 | 1340.50 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 10:15:00 | 1350.20 | 1340.29 | 1340.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 12:15:00 | 1352.25 | 1344.13 | 1341.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 1337.00 | 1347.77 | 1344.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 1337.00 | 1347.77 | 1344.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1337.00 | 1347.77 | 1344.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:30:00 | 1332.90 | 1347.77 | 1344.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1336.45 | 1345.50 | 1344.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:15:00 | 1337.25 | 1345.50 | 1344.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 1331.00 | 1342.60 | 1342.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 12:15:00 | 1327.40 | 1339.56 | 1341.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1301.65 | 1296.53 | 1311.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 1301.65 | 1296.53 | 1311.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1312.85 | 1299.80 | 1311.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 1312.85 | 1299.80 | 1311.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1306.05 | 1301.05 | 1311.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:30:00 | 1302.25 | 1301.05 | 1311.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1314.30 | 1303.40 | 1308.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 1313.25 | 1303.40 | 1308.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1306.15 | 1303.95 | 1308.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 1313.00 | 1303.95 | 1308.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 1305.65 | 1304.29 | 1308.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 1308.30 | 1304.29 | 1308.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 1303.55 | 1304.14 | 1307.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:45:00 | 1298.30 | 1302.43 | 1306.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 1296.60 | 1301.27 | 1305.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 1233.38 | 1268.57 | 1287.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 1231.77 | 1268.57 | 1287.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 1258.15 | 1248.48 | 1264.67 | SL hit (close>ema200) qty=0.50 sl=1248.48 alert=retest2 |

### Cycle 69 — BUY (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 10:15:00 | 1292.65 | 1274.18 | 1272.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 1322.85 | 1297.80 | 1289.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 1329.85 | 1330.14 | 1314.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:00:00 | 1329.85 | 1330.14 | 1314.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1301.50 | 1326.49 | 1320.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 1301.50 | 1326.49 | 1320.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 1312.55 | 1323.71 | 1319.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 1317.50 | 1323.71 | 1319.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 13:45:00 | 1316.35 | 1320.34 | 1319.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:15:00 | 1316.45 | 1320.34 | 1319.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:30:00 | 1323.90 | 1318.93 | 1318.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 1312.20 | 1317.59 | 1318.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 1312.20 | 1317.59 | 1318.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 1305.50 | 1315.41 | 1316.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 1313.75 | 1304.02 | 1309.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 14:15:00 | 1313.75 | 1304.02 | 1309.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 1313.75 | 1304.02 | 1309.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 1313.75 | 1304.02 | 1309.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 1314.80 | 1306.18 | 1309.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 1262.80 | 1306.18 | 1309.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 1199.66 | 1234.49 | 1253.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 1244.45 | 1224.69 | 1237.07 | SL hit (close>ema200) qty=0.50 sl=1224.69 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 1280.30 | 1248.77 | 1246.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1322.10 | 1275.73 | 1260.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 1304.05 | 1314.47 | 1292.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 10:00:00 | 1304.05 | 1314.47 | 1292.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 1274.55 | 1304.89 | 1291.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 1274.55 | 1304.89 | 1291.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 1283.10 | 1300.53 | 1291.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:30:00 | 1290.50 | 1298.43 | 1290.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:30:00 | 1294.70 | 1294.12 | 1290.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:00:00 | 1287.75 | 1293.19 | 1291.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 1273.15 | 1287.40 | 1288.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1273.15 | 1287.40 | 1288.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 1262.70 | 1282.46 | 1286.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 11:15:00 | 1284.75 | 1280.48 | 1284.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 11:15:00 | 1284.75 | 1280.48 | 1284.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1284.75 | 1280.48 | 1284.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 1284.75 | 1280.48 | 1284.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 1266.80 | 1277.74 | 1283.12 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 13:15:00 | 1291.40 | 1285.97 | 1285.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 14:15:00 | 1299.00 | 1288.58 | 1286.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 10:15:00 | 1291.30 | 1291.68 | 1288.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 10:15:00 | 1291.30 | 1291.68 | 1288.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1291.30 | 1291.68 | 1288.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:00:00 | 1291.30 | 1291.68 | 1288.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 1295.40 | 1292.43 | 1289.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:45:00 | 1288.95 | 1292.43 | 1289.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 1290.20 | 1291.98 | 1289.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:00:00 | 1290.20 | 1291.98 | 1289.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 1284.60 | 1290.50 | 1288.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:00:00 | 1284.60 | 1290.50 | 1288.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 1285.65 | 1289.53 | 1288.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 1315.60 | 1288.31 | 1288.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 09:15:00 | 1370.10 | 1373.32 | 1373.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 1370.10 | 1373.32 | 1373.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 1350.85 | 1364.14 | 1368.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 1362.95 | 1358.44 | 1362.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 1362.95 | 1358.44 | 1362.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1362.95 | 1358.44 | 1362.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:45:00 | 1364.30 | 1358.44 | 1362.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1366.00 | 1359.95 | 1363.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1366.00 | 1359.95 | 1363.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1368.55 | 1361.67 | 1363.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1375.10 | 1361.67 | 1363.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1387.90 | 1366.92 | 1365.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 1396.45 | 1388.42 | 1384.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1359.10 | 1384.41 | 1383.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1359.10 | 1384.41 | 1383.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1359.10 | 1384.41 | 1383.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 1363.20 | 1384.41 | 1383.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1363.90 | 1380.30 | 1381.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 1340.20 | 1367.12 | 1375.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 1304.45 | 1304.00 | 1330.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 1304.45 | 1304.00 | 1330.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1295.95 | 1303.40 | 1325.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 1278.30 | 1306.20 | 1318.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 10:15:00 | 1315.10 | 1288.95 | 1288.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 1315.10 | 1288.95 | 1288.38 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 10:15:00 | 1254.00 | 1287.69 | 1290.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 09:15:00 | 1245.20 | 1260.62 | 1273.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 09:15:00 | 1267.10 | 1255.94 | 1263.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 09:15:00 | 1267.10 | 1255.94 | 1263.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1267.10 | 1255.94 | 1263.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:30:00 | 1272.10 | 1255.94 | 1263.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 1278.00 | 1260.35 | 1265.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 11:00:00 | 1278.00 | 1260.35 | 1265.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 1278.70 | 1264.02 | 1266.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 11:45:00 | 1279.60 | 1264.02 | 1266.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 1286.50 | 1268.52 | 1268.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 13:15:00 | 1289.30 | 1272.67 | 1270.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 1318.80 | 1326.44 | 1307.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 1318.80 | 1326.44 | 1307.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1316.00 | 1324.35 | 1307.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1307.00 | 1324.35 | 1307.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 1317.00 | 1320.74 | 1309.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 1307.00 | 1320.74 | 1309.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 1324.30 | 1327.48 | 1320.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 1324.30 | 1327.48 | 1320.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 1325.30 | 1327.05 | 1320.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 1320.00 | 1327.05 | 1320.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1294.30 | 1320.50 | 1318.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1294.30 | 1320.50 | 1318.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1278.40 | 1312.08 | 1314.58 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 1329.40 | 1309.19 | 1308.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 1335.70 | 1316.81 | 1312.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 1330.10 | 1330.67 | 1322.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 13:30:00 | 1328.90 | 1330.67 | 1322.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1338.50 | 1330.92 | 1324.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 11:45:00 | 1352.40 | 1336.55 | 1328.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 14:15:00 | 1350.00 | 1343.52 | 1333.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 14:45:00 | 1350.20 | 1344.35 | 1334.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 1362.40 | 1343.52 | 1334.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 1380.00 | 1390.53 | 1378.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 1380.00 | 1390.53 | 1378.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 1381.50 | 1388.72 | 1379.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:15:00 | 1380.20 | 1388.72 | 1379.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 1377.60 | 1386.50 | 1379.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:15:00 | 1372.60 | 1386.50 | 1379.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 1372.60 | 1383.72 | 1378.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 1385.00 | 1383.72 | 1378.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 1391.00 | 1385.17 | 1379.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:30:00 | 1418.00 | 1391.60 | 1383.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 15:15:00 | 1361.90 | 1389.97 | 1392.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1361.90 | 1389.97 | 1392.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 1352.40 | 1382.46 | 1388.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 1362.40 | 1360.39 | 1374.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 14:00:00 | 1362.40 | 1360.39 | 1374.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1393.20 | 1367.80 | 1374.59 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1398.30 | 1379.42 | 1379.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1410.80 | 1393.03 | 1385.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1389.40 | 1396.04 | 1390.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 1389.40 | 1396.04 | 1390.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1389.40 | 1396.04 | 1390.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 1389.40 | 1396.04 | 1390.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1387.30 | 1394.29 | 1390.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:30:00 | 1388.30 | 1394.29 | 1390.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1390.60 | 1393.55 | 1390.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1397.30 | 1393.28 | 1390.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 12:15:00 | 1384.90 | 1390.59 | 1390.20 | SL hit (close<static) qty=1.00 sl=1385.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 1380.00 | 1388.47 | 1389.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 09:15:00 | 1368.50 | 1383.45 | 1386.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 09:15:00 | 1386.30 | 1376.50 | 1380.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 1386.30 | 1376.50 | 1380.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1386.30 | 1376.50 | 1380.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 1386.30 | 1376.50 | 1380.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1390.00 | 1379.20 | 1381.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 1390.00 | 1379.20 | 1381.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 1398.10 | 1382.98 | 1382.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 1432.40 | 1398.05 | 1390.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 1408.30 | 1409.40 | 1399.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 1408.30 | 1409.40 | 1399.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1405.00 | 1408.52 | 1399.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 1400.50 | 1408.52 | 1399.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1404.20 | 1407.54 | 1400.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1394.00 | 1407.54 | 1400.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1389.60 | 1403.95 | 1399.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:45:00 | 1389.20 | 1403.95 | 1399.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1381.40 | 1399.44 | 1398.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 1382.30 | 1399.44 | 1398.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1380.30 | 1395.61 | 1396.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1367.80 | 1390.05 | 1393.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1391.00 | 1386.14 | 1390.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1391.00 | 1386.14 | 1390.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1391.00 | 1386.14 | 1390.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1390.90 | 1386.14 | 1390.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1381.80 | 1385.27 | 1389.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:30:00 | 1372.70 | 1382.42 | 1388.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 15:15:00 | 1378.00 | 1381.64 | 1386.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 1359.50 | 1346.28 | 1345.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 1359.50 | 1346.28 | 1345.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 1388.50 | 1359.99 | 1352.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 1347.60 | 1371.81 | 1365.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1347.60 | 1371.81 | 1365.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1347.60 | 1371.81 | 1365.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 1352.00 | 1371.81 | 1365.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1340.60 | 1365.57 | 1362.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 1341.00 | 1365.57 | 1362.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 1339.50 | 1360.35 | 1360.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 12:15:00 | 1333.50 | 1354.98 | 1358.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 1342.20 | 1341.56 | 1349.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 1342.20 | 1341.56 | 1349.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1342.20 | 1341.56 | 1349.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 1366.40 | 1341.56 | 1349.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1315.80 | 1307.28 | 1314.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 1315.80 | 1307.28 | 1314.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1322.50 | 1310.33 | 1315.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:30:00 | 1323.80 | 1310.33 | 1315.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1314.90 | 1311.24 | 1315.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 1312.30 | 1310.85 | 1314.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 14:30:00 | 1312.00 | 1310.80 | 1314.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:00:00 | 1310.40 | 1310.80 | 1314.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 1327.60 | 1317.84 | 1316.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 1327.60 | 1317.84 | 1316.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 1380.90 | 1335.90 | 1326.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 1387.40 | 1390.60 | 1370.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:45:00 | 1385.50 | 1390.60 | 1370.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1387.40 | 1398.68 | 1391.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1387.40 | 1398.68 | 1391.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1389.20 | 1396.78 | 1391.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 1385.70 | 1396.78 | 1391.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1371.10 | 1391.65 | 1389.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1371.10 | 1391.65 | 1389.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1370.40 | 1387.40 | 1387.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1328.50 | 1372.80 | 1380.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 1355.10 | 1352.50 | 1363.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 1355.10 | 1352.50 | 1363.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1355.10 | 1352.50 | 1363.76 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 1390.00 | 1372.27 | 1370.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1412.80 | 1383.45 | 1375.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 1425.90 | 1426.06 | 1412.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 15:15:00 | 1421.10 | 1426.06 | 1412.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1412.00 | 1422.46 | 1412.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 1411.30 | 1422.46 | 1412.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1418.20 | 1421.60 | 1413.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 1409.20 | 1421.60 | 1413.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1411.20 | 1419.52 | 1413.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 1409.00 | 1419.52 | 1413.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1397.60 | 1415.14 | 1411.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 1397.60 | 1415.14 | 1411.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 1393.90 | 1407.86 | 1408.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 1388.80 | 1404.05 | 1407.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1407.40 | 1404.47 | 1406.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1407.40 | 1404.47 | 1406.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1407.40 | 1404.47 | 1406.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1407.40 | 1404.47 | 1406.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1402.00 | 1403.98 | 1406.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1395.40 | 1403.98 | 1406.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 1399.00 | 1392.47 | 1396.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 15:00:00 | 1398.90 | 1393.76 | 1397.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1424.50 | 1400.43 | 1399.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1424.50 | 1400.43 | 1399.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 1440.60 | 1421.47 | 1412.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 11:15:00 | 1432.50 | 1436.87 | 1427.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 12:00:00 | 1432.50 | 1436.87 | 1427.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 1427.40 | 1434.98 | 1427.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 1427.30 | 1434.98 | 1427.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 1433.10 | 1434.60 | 1428.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:30:00 | 1437.10 | 1438.02 | 1430.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 10:15:00 | 1580.81 | 1528.60 | 1512.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 1517.50 | 1528.30 | 1529.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 09:15:00 | 1497.50 | 1515.94 | 1522.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 14:15:00 | 1482.30 | 1480.45 | 1493.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 15:00:00 | 1482.30 | 1480.45 | 1493.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1486.80 | 1481.68 | 1492.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:00:00 | 1470.50 | 1480.08 | 1486.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:45:00 | 1471.10 | 1476.53 | 1483.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 1471.80 | 1475.05 | 1480.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 12:30:00 | 1470.70 | 1473.65 | 1478.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1476.00 | 1473.80 | 1477.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 1474.80 | 1473.80 | 1477.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1486.60 | 1476.36 | 1478.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 1486.60 | 1476.36 | 1478.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1473.20 | 1475.73 | 1477.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 1470.00 | 1475.12 | 1477.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 1467.60 | 1474.14 | 1476.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 1501.30 | 1480.78 | 1479.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 1501.30 | 1480.78 | 1479.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 11:15:00 | 1513.40 | 1491.28 | 1484.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 14:15:00 | 1512.60 | 1514.12 | 1504.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 15:00:00 | 1512.60 | 1514.12 | 1504.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1505.70 | 1511.46 | 1505.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 1513.40 | 1511.46 | 1505.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:30:00 | 1513.10 | 1510.51 | 1505.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 1516.00 | 1510.18 | 1506.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:30:00 | 1538.70 | 1510.28 | 1508.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 1486.40 | 1505.50 | 1506.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 1486.40 | 1505.50 | 1506.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 15:15:00 | 1471.30 | 1498.66 | 1502.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 1483.10 | 1481.93 | 1491.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:45:00 | 1484.10 | 1481.93 | 1491.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1402.90 | 1389.39 | 1405.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 1405.90 | 1389.39 | 1405.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1395.40 | 1390.59 | 1404.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 1396.20 | 1390.59 | 1404.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1395.10 | 1390.86 | 1397.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:15:00 | 1401.80 | 1390.86 | 1397.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 1418.90 | 1396.46 | 1399.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:00:00 | 1418.90 | 1396.46 | 1399.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 1418.10 | 1400.79 | 1401.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:15:00 | 1411.20 | 1400.79 | 1401.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 1416.40 | 1403.91 | 1402.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 1416.40 | 1403.91 | 1402.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 15:15:00 | 1422.00 | 1408.89 | 1405.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1382.70 | 1403.65 | 1403.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1382.70 | 1403.65 | 1403.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1382.70 | 1403.65 | 1403.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 1382.70 | 1403.65 | 1403.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1384.90 | 1399.90 | 1401.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 1357.60 | 1386.35 | 1394.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 15:15:00 | 1342.00 | 1332.72 | 1348.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:15:00 | 1328.50 | 1332.72 | 1348.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1304.60 | 1327.10 | 1344.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 1294.60 | 1320.08 | 1339.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 1289.90 | 1300.06 | 1319.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:30:00 | 1295.50 | 1296.54 | 1310.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:00:00 | 1292.90 | 1296.54 | 1310.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1315.30 | 1294.50 | 1299.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 1314.60 | 1303.13 | 1302.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 12:15:00 | 1314.60 | 1303.13 | 1302.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 1343.30 | 1315.37 | 1308.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 1350.40 | 1352.54 | 1342.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:00:00 | 1350.40 | 1352.54 | 1342.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1345.00 | 1349.54 | 1342.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:30:00 | 1346.40 | 1349.54 | 1342.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1343.10 | 1348.25 | 1342.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 1343.10 | 1348.25 | 1342.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1347.90 | 1348.18 | 1343.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 1331.30 | 1348.18 | 1343.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1320.40 | 1342.62 | 1341.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 1318.20 | 1342.62 | 1341.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 1324.50 | 1339.00 | 1339.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 13:15:00 | 1316.90 | 1329.89 | 1334.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 1325.80 | 1322.99 | 1329.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 10:15:00 | 1325.80 | 1322.99 | 1329.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1325.80 | 1322.99 | 1329.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:30:00 | 1326.90 | 1322.99 | 1329.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 1329.50 | 1324.75 | 1328.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:45:00 | 1327.90 | 1324.75 | 1328.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 1337.30 | 1327.26 | 1329.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 1337.30 | 1327.26 | 1329.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 1338.80 | 1329.57 | 1330.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 1328.90 | 1329.57 | 1330.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 1334.70 | 1330.84 | 1330.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 1334.70 | 1330.84 | 1330.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 1335.40 | 1331.75 | 1331.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 15:15:00 | 1334.90 | 1336.16 | 1333.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 15:15:00 | 1334.90 | 1336.16 | 1333.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1334.90 | 1336.16 | 1333.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1342.30 | 1336.16 | 1333.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 12:15:00 | 1333.50 | 1338.09 | 1338.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 1333.50 | 1338.09 | 1338.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 1325.00 | 1332.34 | 1335.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 14:15:00 | 1328.00 | 1326.96 | 1330.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 15:00:00 | 1328.00 | 1326.96 | 1330.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1331.30 | 1327.59 | 1330.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1304.10 | 1326.07 | 1328.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 1310.10 | 1280.48 | 1276.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1310.10 | 1280.48 | 1276.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 10:15:00 | 1326.90 | 1299.90 | 1290.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 1301.10 | 1302.89 | 1293.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:00:00 | 1301.10 | 1302.89 | 1293.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1285.80 | 1298.08 | 1292.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1285.80 | 1298.08 | 1292.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1287.00 | 1295.87 | 1292.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1287.50 | 1295.87 | 1292.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1271.00 | 1287.63 | 1288.86 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 1280.40 | 1279.45 | 1279.41 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 1277.80 | 1279.12 | 1279.27 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 1283.70 | 1280.03 | 1279.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 15:15:00 | 1290.00 | 1282.03 | 1280.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 1299.40 | 1299.47 | 1296.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:30:00 | 1299.30 | 1299.47 | 1296.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1331.60 | 1306.35 | 1300.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 1350.00 | 1306.35 | 1300.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:45:00 | 1335.20 | 1342.32 | 1342.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1334.20 | 1341.11 | 1341.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 1334.20 | 1341.11 | 1341.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 11:15:00 | 1327.50 | 1338.39 | 1340.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 10:15:00 | 1324.40 | 1319.31 | 1328.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 11:00:00 | 1324.40 | 1319.31 | 1328.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 1315.00 | 1320.05 | 1326.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:30:00 | 1328.00 | 1320.05 | 1326.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1257.90 | 1255.35 | 1262.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 1261.40 | 1255.35 | 1262.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1261.60 | 1256.60 | 1262.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1267.20 | 1256.60 | 1262.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1272.00 | 1259.68 | 1263.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 1272.00 | 1259.68 | 1263.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1283.90 | 1264.52 | 1264.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 1283.90 | 1264.52 | 1264.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 1285.90 | 1268.80 | 1266.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 13:15:00 | 1288.10 | 1275.41 | 1270.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 12:15:00 | 1288.80 | 1289.82 | 1281.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 13:00:00 | 1288.80 | 1289.82 | 1281.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1280.40 | 1287.93 | 1281.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 1280.40 | 1287.93 | 1281.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1278.60 | 1286.07 | 1280.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:15:00 | 1277.00 | 1286.07 | 1280.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1277.00 | 1284.25 | 1280.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 1285.40 | 1284.25 | 1280.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:15:00 | 1284.60 | 1280.00 | 1279.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:45:00 | 1279.60 | 1283.61 | 1282.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1279.60 | 1291.49 | 1292.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1279.60 | 1291.49 | 1292.55 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 1296.40 | 1291.40 | 1291.00 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 1286.30 | 1290.59 | 1290.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 1284.00 | 1289.27 | 1290.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 1294.90 | 1288.39 | 1289.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 1294.90 | 1288.39 | 1289.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1294.90 | 1288.39 | 1289.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 1294.90 | 1288.39 | 1289.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 1300.80 | 1290.87 | 1290.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 1307.60 | 1297.95 | 1294.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 11:15:00 | 1301.30 | 1302.04 | 1297.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:45:00 | 1300.80 | 1302.04 | 1297.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1305.90 | 1311.20 | 1306.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:45:00 | 1305.40 | 1311.20 | 1306.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1299.60 | 1308.88 | 1305.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 1299.60 | 1308.88 | 1305.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1295.80 | 1306.26 | 1304.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 1295.00 | 1306.26 | 1304.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 1286.30 | 1302.27 | 1303.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1282.70 | 1295.24 | 1299.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 15:15:00 | 1283.50 | 1282.89 | 1288.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 1291.30 | 1284.57 | 1288.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1291.30 | 1284.57 | 1288.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1286.70 | 1284.57 | 1288.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1289.50 | 1285.56 | 1288.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 1291.90 | 1285.56 | 1288.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1292.70 | 1286.98 | 1288.96 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 1297.90 | 1290.37 | 1290.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 12:15:00 | 1301.30 | 1292.56 | 1291.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 1294.40 | 1296.21 | 1293.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 10:15:00 | 1294.40 | 1296.21 | 1293.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1294.40 | 1296.21 | 1293.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:45:00 | 1293.80 | 1296.21 | 1293.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 1298.20 | 1296.60 | 1294.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 1294.80 | 1296.60 | 1294.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1291.00 | 1298.49 | 1296.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 1291.00 | 1298.49 | 1296.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 11:15:00 | 1283.00 | 1295.39 | 1295.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 1280.50 | 1289.02 | 1292.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1290.40 | 1288.33 | 1291.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1290.40 | 1288.33 | 1291.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1290.40 | 1288.33 | 1291.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 1292.90 | 1288.33 | 1291.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1290.70 | 1288.80 | 1291.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 1295.20 | 1288.80 | 1291.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1287.50 | 1288.54 | 1290.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 1291.10 | 1288.54 | 1290.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1289.80 | 1288.80 | 1290.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:45:00 | 1291.70 | 1288.80 | 1290.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1287.00 | 1288.44 | 1290.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:30:00 | 1291.10 | 1288.44 | 1290.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1291.40 | 1289.03 | 1290.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 1291.40 | 1289.03 | 1290.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1291.00 | 1289.42 | 1290.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1293.50 | 1289.42 | 1290.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1294.80 | 1290.50 | 1290.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 10:15:00 | 1289.70 | 1290.50 | 1290.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 1296.20 | 1291.74 | 1291.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1296.20 | 1291.74 | 1291.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 1299.00 | 1293.19 | 1292.13 | Break + close above crossover candle high |

### Cycle 118 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1271.50 | 1290.16 | 1291.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1251.70 | 1265.26 | 1270.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 09:15:00 | 1224.00 | 1215.41 | 1224.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1224.00 | 1215.41 | 1224.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1224.00 | 1215.41 | 1224.42 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1238.00 | 1227.18 | 1226.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 09:15:00 | 1253.00 | 1236.78 | 1233.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 1238.60 | 1246.33 | 1241.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1238.60 | 1246.33 | 1241.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1238.60 | 1246.33 | 1241.13 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 1235.90 | 1238.69 | 1239.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 1230.30 | 1235.94 | 1237.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 13:15:00 | 1235.00 | 1234.98 | 1236.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 13:15:00 | 1235.00 | 1234.98 | 1236.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1235.00 | 1234.98 | 1236.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:45:00 | 1235.20 | 1234.98 | 1236.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 14:15:00 | 1253.20 | 1238.62 | 1238.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 15:15:00 | 1257.90 | 1242.48 | 1239.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 1238.70 | 1241.95 | 1240.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 1238.70 | 1241.95 | 1240.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1238.70 | 1241.95 | 1240.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 1239.60 | 1241.95 | 1240.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1238.20 | 1241.20 | 1240.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 1237.10 | 1241.20 | 1240.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 1231.10 | 1239.18 | 1239.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 1229.90 | 1237.32 | 1238.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 1229.40 | 1229.34 | 1233.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:45:00 | 1227.70 | 1229.34 | 1233.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 1228.40 | 1229.15 | 1232.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:30:00 | 1227.30 | 1227.84 | 1231.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:30:00 | 1226.30 | 1226.03 | 1227.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:00:00 | 1226.50 | 1226.03 | 1227.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 15:15:00 | 1222.20 | 1217.51 | 1216.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 1222.20 | 1217.51 | 1216.88 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 1210.50 | 1215.40 | 1216.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 1208.10 | 1213.63 | 1214.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 15:15:00 | 1199.00 | 1198.76 | 1203.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 09:15:00 | 1194.00 | 1198.76 | 1203.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1162.00 | 1168.96 | 1175.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 1170.50 | 1168.96 | 1175.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1168.50 | 1168.87 | 1175.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1157.00 | 1169.51 | 1172.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 14:15:00 | 1099.15 | 1116.39 | 1134.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 1142.60 | 1120.29 | 1132.81 | SL hit (close>ema200) qty=0.50 sl=1120.29 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 09:15:00 | 1122.60 | 1118.77 | 1118.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 1131.00 | 1123.36 | 1121.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1156.40 | 1158.26 | 1147.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:15:00 | 1151.00 | 1158.26 | 1147.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1151.40 | 1156.89 | 1147.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 1151.40 | 1156.89 | 1147.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1145.90 | 1154.69 | 1147.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 1145.60 | 1154.69 | 1147.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1137.60 | 1151.27 | 1146.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 1137.60 | 1151.27 | 1146.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 1144.10 | 1147.31 | 1145.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1147.00 | 1147.31 | 1145.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1138.20 | 1145.49 | 1145.12 | SL hit (close<static) qty=1.00 sl=1140.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 1136.50 | 1143.69 | 1144.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 1133.60 | 1141.67 | 1143.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 1136.20 | 1135.38 | 1139.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 1136.20 | 1135.38 | 1139.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1136.20 | 1135.38 | 1139.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 1125.00 | 1132.69 | 1135.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:00:00 | 1125.50 | 1131.25 | 1135.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 13:30:00 | 1125.60 | 1128.50 | 1133.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 1124.10 | 1126.26 | 1130.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1133.60 | 1124.28 | 1127.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 1133.60 | 1124.28 | 1127.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1108.10 | 1121.04 | 1126.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 1139.20 | 1126.76 | 1126.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1139.20 | 1126.76 | 1126.76 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 09:15:00 | 1127.00 | 1128.75 | 1128.85 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1149.30 | 1132.86 | 1130.70 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 1129.40 | 1132.58 | 1132.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 14:15:00 | 1121.50 | 1129.79 | 1131.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 1059.50 | 1057.51 | 1065.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 14:45:00 | 1059.40 | 1057.51 | 1065.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 1058.60 | 1057.73 | 1065.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 1045.60 | 1057.73 | 1065.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1057.10 | 1057.61 | 1064.57 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 1070.30 | 1065.77 | 1065.44 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 15:15:00 | 1061.40 | 1065.33 | 1065.57 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 1071.40 | 1066.54 | 1066.10 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 1057.60 | 1064.88 | 1065.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 1055.30 | 1062.96 | 1064.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 1062.60 | 1058.47 | 1061.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 1062.60 | 1058.47 | 1061.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1062.60 | 1058.47 | 1061.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 1062.60 | 1058.47 | 1061.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1061.30 | 1059.03 | 1061.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 1064.80 | 1059.03 | 1061.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 1063.60 | 1059.95 | 1061.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:30:00 | 1062.90 | 1059.95 | 1061.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 1062.80 | 1060.52 | 1061.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:45:00 | 1063.60 | 1060.52 | 1061.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1062.00 | 1060.99 | 1061.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 1062.00 | 1060.99 | 1061.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1061.70 | 1061.13 | 1061.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1086.00 | 1061.13 | 1061.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 1080.00 | 1064.90 | 1063.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 13:15:00 | 1093.00 | 1079.80 | 1071.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 13:15:00 | 1097.80 | 1098.03 | 1086.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 13:45:00 | 1098.50 | 1098.03 | 1086.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1081.00 | 1095.31 | 1087.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 1087.00 | 1095.31 | 1087.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1088.90 | 1094.02 | 1087.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 14:45:00 | 1103.30 | 1095.10 | 1090.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 12:15:00 | 1079.50 | 1093.85 | 1095.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1079.50 | 1093.85 | 1095.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1075.00 | 1090.08 | 1093.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 12:15:00 | 1041.10 | 1040.99 | 1055.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 13:00:00 | 1041.10 | 1040.99 | 1055.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1064.70 | 1046.96 | 1053.57 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 1058.70 | 1055.82 | 1055.50 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 1053.20 | 1055.48 | 1055.51 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 1062.60 | 1055.87 | 1055.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 1063.00 | 1057.29 | 1056.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 1055.50 | 1056.94 | 1056.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1055.50 | 1056.94 | 1056.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1055.50 | 1056.94 | 1056.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1055.50 | 1056.94 | 1056.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1053.20 | 1056.19 | 1055.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1054.80 | 1056.19 | 1055.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1056.40 | 1056.23 | 1055.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 1054.30 | 1056.23 | 1055.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 1045.70 | 1054.12 | 1055.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 1044.00 | 1052.10 | 1054.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1051.50 | 1044.00 | 1047.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 1051.50 | 1044.00 | 1047.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1051.50 | 1044.00 | 1047.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1051.50 | 1044.00 | 1047.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1048.90 | 1044.98 | 1047.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1059.10 | 1044.98 | 1047.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1055.10 | 1047.00 | 1048.61 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1060.10 | 1051.57 | 1050.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1065.00 | 1055.74 | 1053.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1064.70 | 1068.27 | 1062.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1064.70 | 1068.27 | 1062.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1063.20 | 1067.26 | 1062.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 1063.20 | 1067.26 | 1062.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1089.20 | 1072.14 | 1066.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 1114.80 | 1072.14 | 1066.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 12:15:00 | 1144.00 | 1164.25 | 1164.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 1144.00 | 1164.25 | 1164.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 1140.40 | 1159.48 | 1162.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 11:15:00 | 1107.40 | 1107.11 | 1116.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:30:00 | 1106.00 | 1107.11 | 1116.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1105.50 | 1106.58 | 1112.83 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 1129.80 | 1115.96 | 1115.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 14:15:00 | 1133.90 | 1127.11 | 1124.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 15:15:00 | 1200.00 | 1201.73 | 1192.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:15:00 | 1200.60 | 1201.73 | 1192.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1224.10 | 1219.21 | 1208.86 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 1124.00 | 1192.44 | 1200.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 1109.10 | 1175.77 | 1192.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 12:15:00 | 1045.90 | 1045.12 | 1061.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 13:00:00 | 1045.90 | 1045.12 | 1061.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1050.40 | 1045.52 | 1056.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:30:00 | 1071.10 | 1045.52 | 1056.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 1056.90 | 1048.43 | 1055.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 1059.20 | 1048.43 | 1055.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1046.60 | 1048.07 | 1054.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 1039.00 | 1048.85 | 1053.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 1058.10 | 1050.89 | 1053.69 | SL hit (close>static) qty=1.00 sl=1056.90 alert=retest2 |

### Cycle 145 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 1072.60 | 1058.61 | 1056.93 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 1047.00 | 1054.93 | 1055.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1040.90 | 1052.13 | 1054.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1012.90 | 1012.09 | 1026.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 1017.20 | 1012.09 | 1026.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1020.80 | 1013.34 | 1024.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1020.80 | 1013.34 | 1024.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1015.20 | 1012.68 | 1019.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:00:00 | 1015.20 | 1012.68 | 1019.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 995.10 | 1009.68 | 1017.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 975.00 | 998.07 | 1003.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 926.25 | 961.67 | 981.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 975.00 | 934.30 | 943.14 | SL hit (close>ema200) qty=0.50 sl=934.30 alert=retest2 |

### Cycle 147 — BUY (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 11:15:00 | 954.60 | 945.70 | 945.42 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 939.00 | 945.10 | 945.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 928.00 | 941.68 | 943.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 948.30 | 935.06 | 938.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 948.30 | 935.06 | 938.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 948.30 | 935.06 | 938.08 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 950.80 | 939.96 | 939.89 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 936.45 | 940.66 | 940.85 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 945.80 | 941.69 | 941.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 950.00 | 943.35 | 942.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 957.45 | 960.36 | 954.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 957.45 | 960.36 | 954.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 957.90 | 959.56 | 955.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:00:00 | 957.90 | 959.56 | 955.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 955.50 | 958.75 | 955.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 955.65 | 958.75 | 955.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 956.75 | 958.35 | 955.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:45:00 | 957.45 | 958.35 | 955.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 954.70 | 957.62 | 955.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 976.00 | 957.62 | 955.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 11:15:00 | 1073.60 | 1046.31 | 1020.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 14:15:00 | 1127.50 | 1141.60 | 1142.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 15:15:00 | 1125.00 | 1138.28 | 1140.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1145.35 | 1139.70 | 1140.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1145.35 | 1139.70 | 1140.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1145.35 | 1139.70 | 1140.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1147.85 | 1139.70 | 1140.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1146.30 | 1141.02 | 1141.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1149.60 | 1141.02 | 1141.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1140.95 | 1141.08 | 1141.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:45:00 | 1139.00 | 1140.65 | 1141.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 15:15:00 | 1143.00 | 1141.32 | 1141.38 | SL hit (close>static) qty=1.00 sl=1142.95 alert=retest2 |

### Cycle 153 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1149.60 | 1142.97 | 1142.13 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 1131.50 | 1142.19 | 1142.49 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 1150.35 | 1141.92 | 1141.88 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1120.70 | 1139.03 | 1140.86 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1143.60 | 1139.99 | 1139.61 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 14:15:00 | 1134.30 | 1139.59 | 1139.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 15:15:00 | 1130.20 | 1137.71 | 1138.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 14:15:00 | 1133.50 | 1133.43 | 1135.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 15:00:00 | 1133.50 | 1133.43 | 1135.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 159 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1154.50 | 1137.58 | 1137.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 1162.00 | 1150.61 | 1144.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 1163.50 | 1173.96 | 1163.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 1163.50 | 1173.96 | 1163.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1163.50 | 1173.96 | 1163.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:45:00 | 1165.00 | 1172.37 | 1163.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-21 12:15:00 | 1310.00 | 2024-05-21 13:15:00 | 1301.45 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-05-24 13:00:00 | 1293.25 | 2024-05-27 11:15:00 | 1302.45 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-05-24 14:15:00 | 1291.80 | 2024-05-27 11:15:00 | 1302.45 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-05-27 09:45:00 | 1291.90 | 2024-05-27 11:15:00 | 1302.45 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-06-27 09:30:00 | 1517.40 | 2024-07-01 11:15:00 | 1669.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-28 09:15:00 | 1549.65 | 2024-07-01 12:15:00 | 1704.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-09 10:30:00 | 1658.60 | 2024-07-10 10:15:00 | 1682.15 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-07-09 11:45:00 | 1657.60 | 2024-07-10 10:15:00 | 1682.15 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-07-12 10:30:00 | 1723.65 | 2024-07-19 09:15:00 | 1703.85 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-07-15 10:00:00 | 1738.95 | 2024-07-19 09:15:00 | 1703.85 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-07-18 14:15:00 | 1719.75 | 2024-07-19 09:15:00 | 1703.85 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-07-26 11:45:00 | 1815.40 | 2024-08-01 12:15:00 | 1831.70 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-07-26 12:15:00 | 1822.85 | 2024-08-01 12:15:00 | 1831.70 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-08-05 09:15:00 | 1822.10 | 2024-08-08 10:15:00 | 1814.05 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-08-05 10:15:00 | 1828.25 | 2024-08-08 10:15:00 | 1814.05 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2024-08-06 10:00:00 | 1818.00 | 2024-08-08 10:15:00 | 1814.05 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2024-08-21 13:30:00 | 1861.15 | 2024-08-22 09:15:00 | 1827.00 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-08-21 15:00:00 | 1857.95 | 2024-08-22 09:15:00 | 1827.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-08-27 15:15:00 | 1773.00 | 2024-08-28 10:15:00 | 1792.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-08-28 09:30:00 | 1769.90 | 2024-08-28 10:15:00 | 1792.10 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-09-06 11:15:00 | 1872.15 | 2024-09-06 13:15:00 | 1841.85 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-09-12 13:30:00 | 1822.45 | 2024-09-17 09:15:00 | 1911.00 | STOP_HIT | 1.00 | -4.86% |
| SELL | retest2 | 2024-09-13 09:15:00 | 1817.30 | 2024-09-17 09:15:00 | 1911.00 | STOP_HIT | 1.00 | -5.16% |
| SELL | retest2 | 2024-09-13 11:30:00 | 1819.25 | 2024-09-17 09:15:00 | 1911.00 | STOP_HIT | 1.00 | -5.04% |
| SELL | retest2 | 2024-09-16 13:15:00 | 1823.30 | 2024-09-17 09:15:00 | 1911.00 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest1 | 2024-09-24 10:00:00 | 1945.00 | 2024-09-24 15:15:00 | 1922.10 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest1 | 2024-09-24 11:00:00 | 1939.75 | 2024-09-24 15:15:00 | 1922.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-09-25 11:15:00 | 1984.40 | 2024-09-30 11:15:00 | 1943.55 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-10-09 12:00:00 | 1827.50 | 2024-10-10 09:15:00 | 1878.10 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2024-10-09 14:15:00 | 1828.00 | 2024-10-10 09:15:00 | 1878.10 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-10-15 11:30:00 | 1830.00 | 2024-10-18 09:15:00 | 1647.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-28 13:45:00 | 1179.90 | 2024-12-05 13:15:00 | 1297.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-28 15:00:00 | 1179.65 | 2024-12-05 13:15:00 | 1297.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 10:30:00 | 1179.90 | 2024-12-05 13:15:00 | 1297.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 12:15:00 | 1179.75 | 2024-12-05 13:15:00 | 1297.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 15:15:00 | 1201.45 | 2024-12-12 12:15:00 | 1275.95 | STOP_HIT | 1.00 | 6.20% |
| SELL | retest2 | 2024-12-20 09:15:00 | 1247.65 | 2024-12-24 11:15:00 | 1259.85 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-12-20 11:15:00 | 1256.50 | 2024-12-24 11:15:00 | 1259.85 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-12-20 12:00:00 | 1251.55 | 2024-12-24 11:15:00 | 1259.85 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-12-24 10:30:00 | 1255.25 | 2024-12-24 11:15:00 | 1259.85 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-01-08 12:30:00 | 1247.10 | 2025-01-09 09:15:00 | 1288.50 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-01-08 13:00:00 | 1245.20 | 2025-01-09 09:15:00 | 1288.50 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-01-08 15:15:00 | 1248.00 | 2025-01-09 09:15:00 | 1288.50 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2025-01-10 11:45:00 | 1309.80 | 2025-01-13 10:15:00 | 1249.75 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest2 | 2025-01-24 11:30:00 | 1302.60 | 2025-01-24 13:15:00 | 1279.95 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1367.70 | 2025-02-03 09:15:00 | 1313.90 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2025-02-13 13:45:00 | 1298.30 | 2025-02-14 12:15:00 | 1233.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 1296.60 | 2025-02-14 12:15:00 | 1231.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:45:00 | 1298.30 | 2025-02-17 12:15:00 | 1258.15 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2025-02-13 15:00:00 | 1296.60 | 2025-02-17 12:15:00 | 1258.15 | STOP_HIT | 0.50 | 2.97% |
| BUY | retest2 | 2025-02-24 11:15:00 | 1317.50 | 2025-02-25 10:15:00 | 1312.20 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-02-24 13:45:00 | 1316.35 | 2025-02-25 10:15:00 | 1312.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-02-24 14:15:00 | 1316.45 | 2025-02-25 10:15:00 | 1312.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-02-25 09:30:00 | 1323.90 | 2025-02-25 10:15:00 | 1312.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-02-28 09:15:00 | 1262.80 | 2025-03-04 09:15:00 | 1199.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 1262.80 | 2025-03-05 09:15:00 | 1244.45 | STOP_HIT | 0.50 | 1.45% |
| BUY | retest2 | 2025-03-07 13:30:00 | 1290.50 | 2025-03-10 15:15:00 | 1273.15 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-03-10 09:30:00 | 1294.70 | 2025-03-10 15:15:00 | 1273.15 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-03-10 14:00:00 | 1287.75 | 2025-03-10 15:15:00 | 1273.15 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-03-17 09:15:00 | 1315.60 | 2025-03-26 09:15:00 | 1370.10 | STOP_HIT | 1.00 | 4.14% |
| SELL | retest2 | 2025-04-09 09:45:00 | 1278.30 | 2025-04-15 10:15:00 | 1315.10 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-04-30 11:45:00 | 1352.40 | 2025-05-08 15:15:00 | 1361.90 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2025-04-30 14:15:00 | 1350.00 | 2025-05-08 15:15:00 | 1361.90 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2025-04-30 14:45:00 | 1350.20 | 2025-05-08 15:15:00 | 1361.90 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2025-05-02 09:15:00 | 1362.40 | 2025-05-08 15:15:00 | 1361.90 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-05-07 10:30:00 | 1418.00 | 2025-05-08 15:15:00 | 1361.90 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-05-14 09:15:00 | 1397.30 | 2025-05-14 12:15:00 | 1384.90 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-05-21 11:30:00 | 1372.70 | 2025-05-28 11:15:00 | 1359.50 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-05-21 15:15:00 | 1378.00 | 2025-05-28 11:15:00 | 1359.50 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2025-06-05 12:30:00 | 1312.30 | 2025-06-06 11:15:00 | 1327.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-06-05 14:30:00 | 1312.00 | 2025-06-06 11:15:00 | 1327.60 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-06-05 15:00:00 | 1310.40 | 2025-06-06 11:15:00 | 1327.60 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1395.40 | 2025-06-24 09:15:00 | 1424.50 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-06-23 14:15:00 | 1399.00 | 2025-06-24 09:15:00 | 1424.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-06-23 15:00:00 | 1398.90 | 2025-06-24 09:15:00 | 1424.50 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-06-26 14:30:00 | 1437.10 | 2025-07-04 10:15:00 | 1580.81 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-14 11:00:00 | 1470.50 | 2025-07-17 09:15:00 | 1501.30 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-07-14 12:45:00 | 1471.10 | 2025-07-17 09:15:00 | 1501.30 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-15 09:30:00 | 1471.80 | 2025-07-17 09:15:00 | 1501.30 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-07-15 12:30:00 | 1470.70 | 2025-07-17 09:15:00 | 1501.30 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-07-16 11:30:00 | 1470.00 | 2025-07-17 09:15:00 | 1501.30 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-07-16 12:30:00 | 1467.60 | 2025-07-17 09:15:00 | 1501.30 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-07-21 10:15:00 | 1513.40 | 2025-07-22 14:15:00 | 1486.40 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-07-21 11:30:00 | 1513.10 | 2025-07-22 14:15:00 | 1486.40 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-07-21 15:15:00 | 1516.00 | 2025-07-22 14:15:00 | 1486.40 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-07-22 13:30:00 | 1538.70 | 2025-07-22 14:15:00 | 1486.40 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-07-30 13:15:00 | 1411.20 | 2025-07-30 13:15:00 | 1416.40 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-08-05 10:45:00 | 1294.60 | 2025-08-08 12:15:00 | 1314.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-08-06 09:30:00 | 1289.90 | 2025-08-08 12:15:00 | 1314.60 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-08-06 14:30:00 | 1295.50 | 2025-08-08 12:15:00 | 1314.60 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-08-06 15:00:00 | 1292.90 | 2025-08-08 12:15:00 | 1314.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-08-19 09:15:00 | 1328.90 | 2025-08-19 10:15:00 | 1334.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-08-20 09:15:00 | 1342.30 | 2025-08-21 12:15:00 | 1333.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1304.10 | 2025-09-03 09:15:00 | 1310.10 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-09-16 10:15:00 | 1350.00 | 2025-09-22 10:15:00 | 1334.20 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-19 12:45:00 | 1335.20 | 2025-09-22 10:15:00 | 1334.20 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-10-03 09:15:00 | 1285.40 | 2025-10-08 14:15:00 | 1279.60 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-10-03 14:15:00 | 1284.60 | 2025-10-08 14:15:00 | 1279.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-10-06 14:45:00 | 1279.60 | 2025-10-08 14:15:00 | 1279.60 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-10-29 10:15:00 | 1289.70 | 2025-10-29 11:15:00 | 1296.20 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-11-21 09:30:00 | 1227.30 | 2025-11-26 15:15:00 | 1222.20 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-11-24 12:30:00 | 1226.30 | 2025-11-26 15:15:00 | 1222.20 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-11-24 13:00:00 | 1226.50 | 2025-11-26 15:15:00 | 1222.20 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1157.00 | 2025-12-09 14:15:00 | 1099.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1157.00 | 2025-12-10 09:15:00 | 1142.60 | STOP_HIT | 0.50 | 1.24% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1147.00 | 2025-12-24 09:15:00 | 1138.20 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-29 11:15:00 | 1125.00 | 2025-12-31 12:15:00 | 1139.20 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-12-29 12:00:00 | 1125.50 | 2025-12-31 12:15:00 | 1139.20 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-29 13:30:00 | 1125.60 | 2025-12-31 12:15:00 | 1139.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-30 10:15:00 | 1124.10 | 2025-12-31 12:15:00 | 1139.20 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-01-21 14:45:00 | 1103.30 | 2026-01-23 12:15:00 | 1079.50 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-02-06 10:15:00 | 1114.80 | 2026-02-12 12:15:00 | 1144.00 | STOP_HIT | 1.00 | 2.62% |
| SELL | retest2 | 2026-03-12 09:15:00 | 1039.00 | 2026-03-12 10:15:00 | 1058.10 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-03-20 12:15:00 | 975.00 | 2026-03-23 10:15:00 | 926.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 975.00 | 2026-03-25 09:15:00 | 975.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 976.00 | 2026-04-10 11:15:00 | 1073.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 13:45:00 | 1139.00 | 2026-04-27 15:15:00 | 1143.00 | STOP_HIT | 1.00 | -0.35% |
