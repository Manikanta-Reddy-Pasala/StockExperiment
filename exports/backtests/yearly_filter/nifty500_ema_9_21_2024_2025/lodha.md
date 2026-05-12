# Lodha Developers Ltd. (LODHA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 960.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 129 |
| ALERT1 | 88 |
| ALERT2 | 86 |
| ALERT2_SKIP | 45 |
| ALERT3 | 218 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 106 |
| PARTIAL | 23 |
| TARGET_HIT | 12 |
| STOP_HIT | 99 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 134 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 76 / 58
- **Target hits / Stop hits / Partials:** 12 / 99 / 23
- **Avg / median % per leg:** 1.58% / 0.74%
- **Sum % (uncompounded):** 212.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 18 | 40.0% | 3 | 42 | 0 | -0.06% | -2.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 45 | 18 | 40.0% | 3 | 42 | 0 | -0.06% | -2.6% |
| SELL (all) | 89 | 58 | 65.2% | 9 | 57 | 23 | 2.41% | 214.7% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 6 | 0 | -0.94% | -5.6% |
| SELL @ 3rd Alert (retest2) | 83 | 54 | 65.1% | 9 | 51 | 23 | 2.66% | 220.4% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 6 | 0 | -0.94% | -5.6% |
| retest2 (combined) | 128 | 72 | 56.2% | 12 | 93 | 23 | 1.70% | 217.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 1140.00 | 1123.95 | 1123.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 1162.35 | 1131.63 | 1126.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 13:15:00 | 1195.35 | 1197.52 | 1187.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 13:30:00 | 1192.80 | 1197.52 | 1187.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 1345.60 | 1348.12 | 1339.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:45:00 | 1337.30 | 1348.12 | 1339.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 1345.00 | 1347.50 | 1339.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 1303.60 | 1347.50 | 1339.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1298.20 | 1337.64 | 1336.17 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 1297.55 | 1329.62 | 1332.66 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 09:15:00 | 1340.15 | 1317.92 | 1317.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 1384.60 | 1347.27 | 1333.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 1383.45 | 1410.04 | 1386.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 1383.45 | 1410.04 | 1386.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1383.45 | 1410.04 | 1386.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:45:00 | 1388.85 | 1410.04 | 1386.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1340.90 | 1396.21 | 1382.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 1340.90 | 1396.21 | 1382.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1350.00 | 1386.97 | 1379.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 14:00:00 | 1361.70 | 1381.92 | 1377.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 14:15:00 | 1313.90 | 1368.31 | 1372.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 14:15:00 | 1313.90 | 1368.31 | 1372.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 1260.95 | 1337.48 | 1356.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 1328.90 | 1320.31 | 1342.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 1328.90 | 1320.31 | 1342.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1403.30 | 1331.46 | 1340.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 1423.95 | 1331.46 | 1340.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1399.00 | 1344.97 | 1345.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 11:15:00 | 1386.00 | 1344.97 | 1345.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 11:15:00 | 1375.55 | 1351.09 | 1348.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 1375.55 | 1351.09 | 1348.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1424.05 | 1381.24 | 1365.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 10:15:00 | 1474.40 | 1477.61 | 1460.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 10:45:00 | 1470.60 | 1477.61 | 1460.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1530.00 | 1574.89 | 1558.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 1530.00 | 1574.89 | 1558.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1525.00 | 1564.91 | 1555.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 1515.00 | 1564.91 | 1555.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 1541.15 | 1549.38 | 1549.67 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 1573.20 | 1553.49 | 1551.41 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 11:15:00 | 1536.20 | 1548.96 | 1549.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 12:15:00 | 1529.05 | 1544.98 | 1547.77 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 14:15:00 | 1577.15 | 1550.48 | 1549.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 15:15:00 | 1584.95 | 1557.38 | 1552.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 1555.40 | 1576.68 | 1568.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 1555.40 | 1576.68 | 1568.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1555.40 | 1576.68 | 1568.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 1551.45 | 1576.68 | 1568.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1552.55 | 1571.85 | 1567.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 1552.55 | 1571.85 | 1567.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 1546.70 | 1561.12 | 1563.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 1510.05 | 1547.81 | 1556.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 1458.50 | 1454.60 | 1481.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 09:45:00 | 1457.45 | 1454.60 | 1481.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1479.00 | 1460.16 | 1477.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 1479.00 | 1460.16 | 1477.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1506.40 | 1469.41 | 1480.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 1506.40 | 1469.41 | 1480.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1542.45 | 1484.01 | 1485.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 1542.45 | 1484.01 | 1485.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 1548.00 | 1496.81 | 1491.58 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 13:15:00 | 1494.20 | 1509.72 | 1511.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 14:15:00 | 1491.30 | 1506.04 | 1509.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 14:15:00 | 1493.90 | 1491.18 | 1498.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-03 15:00:00 | 1493.90 | 1491.18 | 1498.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 1494.00 | 1491.74 | 1498.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 1493.30 | 1491.74 | 1498.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1490.35 | 1491.47 | 1497.46 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 13:15:00 | 1516.60 | 1501.79 | 1500.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 09:15:00 | 1526.05 | 1508.17 | 1504.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 14:15:00 | 1533.15 | 1537.17 | 1526.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 15:00:00 | 1533.15 | 1537.17 | 1526.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1516.30 | 1544.88 | 1539.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 1516.20 | 1544.88 | 1539.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1523.15 | 1540.54 | 1538.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 13:00:00 | 1532.85 | 1539.00 | 1537.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 13:15:00 | 1522.65 | 1535.73 | 1536.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 1522.65 | 1535.73 | 1536.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 09:15:00 | 1480.20 | 1520.95 | 1529.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 10:15:00 | 1448.65 | 1422.16 | 1449.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 10:15:00 | 1448.65 | 1422.16 | 1449.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 1448.65 | 1422.16 | 1449.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:00:00 | 1448.65 | 1422.16 | 1449.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 1457.95 | 1429.32 | 1450.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:00:00 | 1457.95 | 1429.32 | 1450.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 1484.45 | 1440.34 | 1453.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:30:00 | 1492.90 | 1440.34 | 1453.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1463.00 | 1455.63 | 1458.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 1454.15 | 1455.63 | 1458.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1474.00 | 1459.31 | 1459.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 1474.00 | 1459.31 | 1459.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 1468.55 | 1461.15 | 1460.62 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 11:15:00 | 1454.00 | 1459.72 | 1460.02 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 12:15:00 | 1465.45 | 1460.87 | 1460.51 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 1432.25 | 1455.02 | 1457.99 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 15:15:00 | 1492.05 | 1458.60 | 1456.75 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 1436.05 | 1454.09 | 1454.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 1419.85 | 1447.24 | 1451.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 13:15:00 | 1439.05 | 1438.93 | 1446.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 13:15:00 | 1439.05 | 1438.93 | 1446.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 1439.05 | 1438.93 | 1446.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:00:00 | 1439.05 | 1438.93 | 1446.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 1444.70 | 1440.08 | 1445.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 15:00:00 | 1444.70 | 1440.08 | 1445.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 1435.00 | 1439.07 | 1444.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 09:15:00 | 1422.85 | 1439.07 | 1444.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 11:00:00 | 1427.65 | 1436.10 | 1442.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:15:00 | 1432.80 | 1436.03 | 1441.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 13:00:00 | 1432.80 | 1435.38 | 1441.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1438.00 | 1435.91 | 1440.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:00:00 | 1438.00 | 1435.91 | 1440.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 1441.40 | 1437.01 | 1440.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 1441.40 | 1437.01 | 1440.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 1427.00 | 1435.00 | 1439.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 1440.45 | 1435.00 | 1439.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1441.80 | 1436.36 | 1439.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 1401.05 | 1440.48 | 1441.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 15:15:00 | 1361.16 | 1407.54 | 1423.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 15:15:00 | 1361.16 | 1407.54 | 1423.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 14:15:00 | 1404.30 | 1400.64 | 1412.57 | SL hit (close>ema200) qty=0.50 sl=1400.64 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 1220.00 | 1211.03 | 1209.90 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 10:15:00 | 1199.40 | 1208.61 | 1208.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 11:15:00 | 1195.60 | 1206.00 | 1207.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 1228.70 | 1201.86 | 1203.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 1228.70 | 1201.86 | 1203.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1228.70 | 1201.86 | 1203.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:30:00 | 1234.25 | 1201.86 | 1203.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 1234.05 | 1208.29 | 1206.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 12:15:00 | 1244.95 | 1220.11 | 1212.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 1307.85 | 1308.40 | 1284.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 15:00:00 | 1307.85 | 1308.40 | 1284.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1291.80 | 1304.78 | 1287.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1285.75 | 1304.78 | 1287.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1291.15 | 1302.06 | 1287.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:45:00 | 1286.50 | 1302.06 | 1287.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 1299.15 | 1301.47 | 1288.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:30:00 | 1310.00 | 1295.00 | 1289.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 13:45:00 | 1307.90 | 1296.40 | 1291.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 10:15:00 | 1283.40 | 1293.50 | 1291.98 | SL hit (close<static) qty=1.00 sl=1288.55 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 12:15:00 | 1281.85 | 1289.19 | 1290.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 1258.05 | 1281.26 | 1285.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 11:15:00 | 1184.10 | 1177.49 | 1198.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 12:00:00 | 1184.10 | 1177.49 | 1198.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 1200.05 | 1182.01 | 1199.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:45:00 | 1197.20 | 1182.01 | 1199.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 1218.70 | 1189.34 | 1200.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:00:00 | 1218.70 | 1189.34 | 1200.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 1213.25 | 1194.13 | 1201.94 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 1261.00 | 1211.16 | 1208.56 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 1242.00 | 1248.75 | 1248.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 14:15:00 | 1222.90 | 1241.80 | 1245.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 15:15:00 | 1224.00 | 1221.76 | 1230.78 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 09:15:00 | 1208.75 | 1221.76 | 1230.78 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 11:00:00 | 1207.45 | 1216.55 | 1226.70 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 12:15:00 | 1207.85 | 1215.47 | 1225.28 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 13:00:00 | 1207.55 | 1213.88 | 1223.67 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1196.00 | 1194.60 | 1204.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 12:30:00 | 1183.35 | 1194.57 | 1201.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 15:00:00 | 1185.70 | 1191.91 | 1196.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 1183.00 | 1191.13 | 1195.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:30:00 | 1183.60 | 1191.29 | 1194.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 1202.25 | 1193.48 | 1195.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 1202.25 | 1193.48 | 1195.21 | SL hit (close>ema400) qty=1.00 sl=1195.21 alert=retest1 |

### Cycle 27 — BUY (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 15:15:00 | 1204.00 | 1195.92 | 1195.91 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 1191.55 | 1195.05 | 1195.51 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 1221.00 | 1199.15 | 1196.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 1285.05 | 1232.94 | 1216.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 1260.75 | 1261.57 | 1242.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 1260.75 | 1261.57 | 1242.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1260.75 | 1261.57 | 1242.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:30:00 | 1246.80 | 1261.57 | 1242.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1268.15 | 1283.26 | 1275.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 12:00:00 | 1268.15 | 1283.26 | 1275.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 1272.20 | 1281.05 | 1274.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 14:45:00 | 1291.30 | 1286.06 | 1278.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-23 09:15:00 | 1420.43 | 1368.95 | 1332.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 11:15:00 | 1356.85 | 1377.82 | 1379.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 12:15:00 | 1332.55 | 1363.45 | 1370.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 12:15:00 | 1174.85 | 1173.09 | 1203.75 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-04 13:30:00 | 1163.55 | 1170.90 | 1199.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-04 14:00:00 | 1162.15 | 1170.90 | 1199.97 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1206.50 | 1179.25 | 1196.66 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 1206.50 | 1179.25 | 1196.66 | SL hit (close>ema400) qty=1.00 sl=1196.66 alert=retest1 |

### Cycle 31 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1209.10 | 1191.93 | 1189.63 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 1183.00 | 1193.54 | 1194.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 1179.60 | 1189.07 | 1192.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 1183.50 | 1177.17 | 1182.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 1183.50 | 1177.17 | 1182.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1183.50 | 1177.17 | 1182.96 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 1200.45 | 1188.91 | 1187.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 14:15:00 | 1210.50 | 1195.63 | 1190.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 1197.50 | 1211.37 | 1203.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 1197.50 | 1211.37 | 1203.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1197.50 | 1211.37 | 1203.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:45:00 | 1198.95 | 1211.37 | 1203.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 1208.10 | 1210.72 | 1204.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:45:00 | 1193.65 | 1210.72 | 1204.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 1208.65 | 1210.30 | 1204.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 1203.25 | 1210.30 | 1204.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1183.80 | 1209.31 | 1206.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1183.80 | 1209.31 | 1206.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 1180.65 | 1203.58 | 1204.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 1174.65 | 1197.79 | 1201.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 1161.80 | 1161.68 | 1175.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 14:15:00 | 1161.80 | 1161.68 | 1175.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1161.80 | 1161.68 | 1175.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 1172.00 | 1161.68 | 1175.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1156.40 | 1160.36 | 1172.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:00:00 | 1149.00 | 1156.58 | 1167.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:30:00 | 1147.75 | 1154.77 | 1165.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 1091.55 | 1117.54 | 1139.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 1090.36 | 1117.54 | 1139.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 14:15:00 | 1087.10 | 1085.39 | 1107.16 | SL hit (close>ema200) qty=0.50 sl=1085.39 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 1107.35 | 1081.16 | 1079.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 14:15:00 | 1112.50 | 1090.78 | 1084.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 10:15:00 | 1197.55 | 1200.05 | 1183.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 14:15:00 | 1191.50 | 1198.93 | 1188.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 1191.50 | 1198.93 | 1188.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 1191.50 | 1198.93 | 1188.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 1195.00 | 1198.15 | 1188.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 1174.20 | 1198.15 | 1188.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1170.45 | 1192.61 | 1187.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:45:00 | 1169.70 | 1192.61 | 1187.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 1163.25 | 1186.74 | 1185.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 1163.25 | 1186.74 | 1185.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 11:15:00 | 1160.75 | 1181.54 | 1182.90 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 1213.80 | 1181.65 | 1180.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 1232.70 | 1208.22 | 1195.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 09:15:00 | 1197.90 | 1212.15 | 1204.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 1197.90 | 1212.15 | 1204.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1197.90 | 1212.15 | 1204.92 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 1183.70 | 1198.96 | 1200.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 1165.10 | 1189.72 | 1195.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 1195.30 | 1188.49 | 1193.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 1195.30 | 1188.49 | 1193.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1195.30 | 1188.49 | 1193.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:00:00 | 1195.30 | 1188.49 | 1193.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 1192.35 | 1189.26 | 1193.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:30:00 | 1201.15 | 1189.26 | 1193.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1189.85 | 1189.38 | 1193.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:00:00 | 1187.05 | 1189.24 | 1192.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 1277.25 | 1206.96 | 1199.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 1277.25 | 1206.96 | 1199.84 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 14:15:00 | 1200.00 | 1213.57 | 1214.85 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 09:15:00 | 1232.55 | 1216.48 | 1215.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 09:15:00 | 1257.10 | 1227.85 | 1222.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 14:15:00 | 1230.30 | 1232.46 | 1227.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 14:15:00 | 1230.30 | 1232.46 | 1227.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1230.30 | 1232.46 | 1227.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:45:00 | 1219.00 | 1232.46 | 1227.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 1222.00 | 1230.37 | 1226.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 1283.50 | 1230.37 | 1226.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 1280.00 | 1248.33 | 1247.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 15:15:00 | 1249.50 | 1255.73 | 1253.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 14:15:00 | 1219.45 | 1251.27 | 1253.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 14:15:00 | 1219.45 | 1251.27 | 1253.05 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 09:15:00 | 1296.90 | 1257.43 | 1255.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 1308.45 | 1289.00 | 1279.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 1286.70 | 1290.58 | 1282.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 1286.70 | 1290.58 | 1282.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1274.80 | 1288.09 | 1283.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 1274.80 | 1288.09 | 1283.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 1277.70 | 1286.01 | 1282.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:30:00 | 1298.60 | 1285.62 | 1282.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 13:15:00 | 1272.85 | 1279.74 | 1280.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 13:15:00 | 1272.85 | 1279.74 | 1280.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 14:15:00 | 1250.80 | 1273.96 | 1277.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 1289.15 | 1272.52 | 1276.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 1289.15 | 1272.52 | 1276.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1289.15 | 1272.52 | 1276.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:45:00 | 1295.00 | 1272.52 | 1276.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 1284.95 | 1275.01 | 1277.10 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 1294.15 | 1278.84 | 1278.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 1306.00 | 1291.43 | 1285.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 12:15:00 | 1288.85 | 1291.65 | 1287.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 12:15:00 | 1288.85 | 1291.65 | 1287.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 1288.85 | 1291.65 | 1287.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 15:00:00 | 1295.90 | 1292.62 | 1288.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-12 10:15:00 | 1425.49 | 1408.51 | 1396.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 11:15:00 | 1458.20 | 1468.13 | 1468.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 1436.85 | 1455.93 | 1461.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 1454.60 | 1435.73 | 1448.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 1454.60 | 1435.73 | 1448.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1454.60 | 1435.73 | 1448.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 1458.00 | 1435.73 | 1448.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1446.35 | 1437.85 | 1448.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 12:45:00 | 1435.90 | 1440.04 | 1447.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 09:15:00 | 1411.05 | 1441.24 | 1446.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 09:15:00 | 1364.11 | 1401.62 | 1409.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 13:15:00 | 1387.95 | 1385.35 | 1397.85 | SL hit (close>ema200) qty=0.50 sl=1385.35 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 12:15:00 | 1392.80 | 1380.51 | 1379.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 15:15:00 | 1403.95 | 1389.84 | 1384.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 1364.70 | 1386.53 | 1383.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 10:15:00 | 1364.70 | 1386.53 | 1383.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1364.70 | 1386.53 | 1383.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1364.70 | 1386.53 | 1383.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 1365.00 | 1382.22 | 1382.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1354.70 | 1373.78 | 1378.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 1348.80 | 1344.64 | 1355.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 1348.80 | 1344.64 | 1355.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1348.80 | 1344.64 | 1355.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:30:00 | 1324.00 | 1342.89 | 1348.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:45:00 | 1325.40 | 1338.40 | 1345.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:15:00 | 1324.00 | 1338.40 | 1345.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1257.80 | 1282.37 | 1305.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1259.13 | 1282.37 | 1305.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1257.80 | 1282.37 | 1305.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 12:15:00 | 1191.60 | 1239.90 | 1278.50 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 13:15:00 | 1175.70 | 1162.69 | 1161.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 14:15:00 | 1177.70 | 1165.70 | 1162.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1168.55 | 1192.65 | 1182.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 1168.55 | 1192.65 | 1182.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1168.55 | 1192.65 | 1182.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1168.55 | 1192.65 | 1182.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1144.00 | 1182.92 | 1179.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 1144.00 | 1182.92 | 1179.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 1162.25 | 1174.72 | 1175.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 1146.00 | 1166.15 | 1171.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 10:15:00 | 1104.95 | 1088.91 | 1103.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 10:15:00 | 1104.95 | 1088.91 | 1103.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1104.95 | 1088.91 | 1103.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 1104.95 | 1088.91 | 1103.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 1100.00 | 1091.13 | 1102.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 1091.55 | 1093.90 | 1103.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 1096.45 | 1095.82 | 1102.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 11:15:00 | 1119.95 | 1103.78 | 1104.26 | SL hit (close>static) qty=1.00 sl=1109.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-27 12:15:00 | 1123.35 | 1107.69 | 1106.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 11:15:00 | 1139.00 | 1118.19 | 1112.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-28 15:15:00 | 1127.00 | 1128.47 | 1119.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:15:00 | 1107.70 | 1128.47 | 1119.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1113.00 | 1125.38 | 1119.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 09:15:00 | 1152.05 | 1126.68 | 1121.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-01 14:15:00 | 1267.26 | 1231.72 | 1202.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 1224.10 | 1266.58 | 1267.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 1213.25 | 1239.97 | 1252.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 1227.65 | 1225.79 | 1242.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 09:45:00 | 1220.00 | 1225.79 | 1242.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1232.90 | 1228.95 | 1240.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:30:00 | 1242.35 | 1228.95 | 1240.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 1238.55 | 1229.87 | 1238.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 1238.55 | 1229.87 | 1238.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 1235.95 | 1231.08 | 1237.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 1203.90 | 1231.08 | 1237.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1184.00 | 1221.67 | 1233.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:00:00 | 1183.75 | 1214.08 | 1228.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 11:00:00 | 1183.70 | 1171.62 | 1184.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:45:00 | 1179.25 | 1176.20 | 1180.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1124.56 | 1155.11 | 1165.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1124.52 | 1155.11 | 1165.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 11:15:00 | 1159.35 | 1154.84 | 1163.38 | SL hit (close>ema200) qty=0.50 sl=1154.84 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 12:15:00 | 1170.50 | 1165.79 | 1165.45 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 14:15:00 | 1160.25 | 1165.04 | 1165.19 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 1194.40 | 1170.80 | 1167.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 09:15:00 | 1202.90 | 1190.94 | 1184.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 11:15:00 | 1205.00 | 1206.88 | 1199.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 12:00:00 | 1205.00 | 1206.88 | 1199.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1196.30 | 1204.76 | 1198.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 1196.30 | 1204.76 | 1198.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 1195.75 | 1202.96 | 1198.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:45:00 | 1213.30 | 1202.25 | 1199.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 1211.35 | 1198.42 | 1198.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 11:00:00 | 1201.25 | 1200.89 | 1199.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 11:15:00 | 1191.00 | 1198.92 | 1198.71 | SL hit (close<static) qty=1.00 sl=1193.15 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 1185.75 | 1196.28 | 1197.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 1154.10 | 1182.08 | 1189.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 13:15:00 | 1125.95 | 1122.66 | 1137.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 14:00:00 | 1125.95 | 1122.66 | 1137.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1140.15 | 1124.35 | 1134.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 1144.55 | 1124.35 | 1134.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1146.25 | 1128.73 | 1135.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 1146.75 | 1128.73 | 1135.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 1148.90 | 1139.59 | 1139.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1165.30 | 1149.08 | 1144.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 12:15:00 | 1149.60 | 1153.66 | 1147.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 13:00:00 | 1149.60 | 1153.66 | 1147.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 1143.30 | 1151.59 | 1147.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:45:00 | 1142.90 | 1151.59 | 1147.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 1139.45 | 1149.16 | 1146.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 15:00:00 | 1139.45 | 1149.16 | 1146.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 09:15:00 | 1131.70 | 1143.40 | 1144.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 1111.40 | 1135.38 | 1140.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 1116.40 | 1097.96 | 1111.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 1116.40 | 1097.96 | 1111.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1116.40 | 1097.96 | 1111.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:00:00 | 1116.40 | 1097.96 | 1111.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1130.00 | 1104.37 | 1112.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 1130.00 | 1104.37 | 1112.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1133.15 | 1110.12 | 1114.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 1133.15 | 1110.12 | 1114.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 1140.00 | 1119.16 | 1118.13 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 1099.95 | 1115.90 | 1117.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 1093.35 | 1111.39 | 1115.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 11:15:00 | 1070.45 | 1067.57 | 1080.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 11:45:00 | 1072.75 | 1067.57 | 1080.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1079.65 | 1069.08 | 1076.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 1079.65 | 1069.08 | 1076.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1069.20 | 1069.10 | 1075.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 11:45:00 | 1064.80 | 1067.65 | 1074.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:30:00 | 1067.95 | 1069.43 | 1074.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 13:15:00 | 1101.90 | 1075.93 | 1077.04 | SL hit (close>static) qty=1.00 sl=1079.50 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 1113.65 | 1083.47 | 1080.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1137.90 | 1098.94 | 1088.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 1189.00 | 1190.40 | 1161.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 1217.85 | 1190.40 | 1161.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 1170.75 | 1186.38 | 1167.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:45:00 | 1173.85 | 1186.38 | 1167.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 1166.95 | 1179.63 | 1167.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 1166.90 | 1179.63 | 1167.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 1181.80 | 1180.06 | 1168.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 15:15:00 | 1188.00 | 1180.06 | 1168.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 15:15:00 | 1196.90 | 1210.97 | 1212.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 1196.90 | 1210.97 | 1212.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 09:15:00 | 1172.20 | 1203.21 | 1209.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 1176.15 | 1171.11 | 1185.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 1176.15 | 1171.11 | 1185.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1176.15 | 1171.11 | 1185.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 1193.20 | 1171.11 | 1185.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1198.00 | 1176.49 | 1186.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 1198.00 | 1176.49 | 1186.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1194.20 | 1180.03 | 1187.48 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 1219.60 | 1196.24 | 1193.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 11:15:00 | 1221.80 | 1209.61 | 1201.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 15:15:00 | 1214.80 | 1214.91 | 1207.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 09:15:00 | 1202.50 | 1214.91 | 1207.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1187.05 | 1209.34 | 1205.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1187.05 | 1209.34 | 1205.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1198.70 | 1207.21 | 1204.73 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 1176.80 | 1198.26 | 1200.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 1171.30 | 1192.87 | 1198.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 1137.00 | 1131.19 | 1153.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 1137.00 | 1131.19 | 1153.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1114.75 | 1116.32 | 1128.60 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 1187.00 | 1135.59 | 1132.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 1204.00 | 1173.56 | 1153.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 11:15:00 | 1230.00 | 1233.02 | 1210.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 12:00:00 | 1230.00 | 1233.02 | 1210.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1313.10 | 1338.54 | 1319.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:45:00 | 1316.70 | 1338.54 | 1319.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 1316.30 | 1334.09 | 1319.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:30:00 | 1311.60 | 1334.09 | 1319.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1320.40 | 1331.35 | 1319.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 1382.50 | 1324.46 | 1318.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:00:00 | 1334.10 | 1326.39 | 1320.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 13:15:00 | 1300.30 | 1316.94 | 1317.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 1300.30 | 1316.94 | 1317.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 1291.10 | 1307.64 | 1312.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 14:15:00 | 1308.30 | 1299.82 | 1305.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 14:15:00 | 1308.30 | 1299.82 | 1305.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 1308.30 | 1299.82 | 1305.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 1308.30 | 1299.82 | 1305.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 1313.00 | 1302.46 | 1306.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 1304.70 | 1302.46 | 1306.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 1353.60 | 1306.02 | 1304.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 1353.60 | 1306.02 | 1304.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 09:15:00 | 1376.60 | 1338.72 | 1325.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 13:15:00 | 1343.60 | 1345.18 | 1333.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 14:00:00 | 1343.60 | 1345.18 | 1333.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1324.50 | 1340.17 | 1333.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:30:00 | 1325.10 | 1340.17 | 1333.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1324.10 | 1336.95 | 1332.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 1324.10 | 1336.95 | 1332.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1324.20 | 1334.40 | 1332.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:45:00 | 1324.60 | 1334.40 | 1332.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 1332.40 | 1333.57 | 1332.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:30:00 | 1329.50 | 1333.57 | 1332.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1332.30 | 1333.32 | 1332.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:30:00 | 1327.80 | 1333.32 | 1332.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 1332.00 | 1333.05 | 1332.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 1324.40 | 1333.05 | 1332.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 1317.60 | 1329.96 | 1330.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 1295.90 | 1317.68 | 1324.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 1301.00 | 1298.07 | 1307.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 15:00:00 | 1301.00 | 1298.07 | 1307.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1306.50 | 1300.06 | 1307.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 1293.70 | 1300.15 | 1305.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1229.01 | 1284.71 | 1296.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 1292.30 | 1256.92 | 1271.87 | SL hit (close>ema200) qty=0.50 sl=1256.92 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1306.00 | 1279.27 | 1278.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 1307.90 | 1288.98 | 1283.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 1272.70 | 1285.73 | 1282.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 1272.70 | 1285.73 | 1282.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 1272.70 | 1285.73 | 1282.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:00:00 | 1272.70 | 1285.73 | 1282.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1271.60 | 1282.90 | 1281.67 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 1268.50 | 1278.98 | 1280.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 13:15:00 | 1265.50 | 1276.28 | 1278.71 | Break + close below crossover candle low |

### Cycle 71 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 1313.50 | 1282.99 | 1281.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 10:15:00 | 1323.00 | 1290.99 | 1284.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1408.80 | 1413.71 | 1392.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1408.80 | 1413.71 | 1392.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1408.80 | 1413.71 | 1392.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1393.50 | 1413.71 | 1392.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 1394.50 | 1407.69 | 1396.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 1394.50 | 1407.69 | 1396.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1380.60 | 1402.27 | 1394.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 1380.60 | 1402.27 | 1394.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 1385.00 | 1398.82 | 1394.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 1404.10 | 1398.82 | 1394.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1437.80 | 1450.25 | 1450.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 1437.80 | 1450.25 | 1450.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 1433.30 | 1446.86 | 1448.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 11:15:00 | 1434.80 | 1431.04 | 1437.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 11:15:00 | 1434.80 | 1431.04 | 1437.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1434.80 | 1431.04 | 1437.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:00:00 | 1434.80 | 1431.04 | 1437.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1439.40 | 1432.71 | 1437.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:30:00 | 1442.50 | 1432.71 | 1437.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1435.70 | 1433.31 | 1437.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 1435.70 | 1433.31 | 1437.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1433.00 | 1433.25 | 1437.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 1430.20 | 1433.25 | 1437.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 1457.40 | 1437.59 | 1438.51 | SL hit (close>static) qty=1.00 sl=1438.80 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 1455.10 | 1441.09 | 1440.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 11:15:00 | 1470.50 | 1446.97 | 1442.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 1447.10 | 1448.42 | 1444.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 15:00:00 | 1447.10 | 1448.42 | 1444.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 1447.50 | 1448.23 | 1444.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1452.60 | 1448.23 | 1444.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 1451.20 | 1448.83 | 1445.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:45:00 | 1453.80 | 1450.16 | 1446.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 1463.00 | 1447.38 | 1446.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1459.70 | 1458.04 | 1453.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 1471.00 | 1458.67 | 1453.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 13:15:00 | 1469.70 | 1489.56 | 1490.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 1469.70 | 1489.56 | 1490.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 14:15:00 | 1468.20 | 1485.28 | 1488.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1485.10 | 1482.80 | 1486.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1485.10 | 1482.80 | 1486.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1485.10 | 1482.80 | 1486.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 1487.40 | 1482.80 | 1486.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1475.00 | 1476.78 | 1481.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 1466.00 | 1476.78 | 1481.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 1470.10 | 1458.82 | 1458.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 1470.10 | 1458.82 | 1458.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 1472.90 | 1461.64 | 1459.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 1464.80 | 1466.90 | 1463.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 11:15:00 | 1464.80 | 1466.90 | 1463.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1464.80 | 1466.90 | 1463.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 1464.80 | 1466.90 | 1463.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 1460.30 | 1465.58 | 1462.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 1460.40 | 1465.58 | 1462.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 1451.90 | 1462.85 | 1461.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:45:00 | 1447.40 | 1462.85 | 1461.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 1447.60 | 1459.80 | 1460.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 1440.30 | 1449.64 | 1454.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 1450.60 | 1447.38 | 1451.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 1450.60 | 1447.38 | 1451.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1450.60 | 1447.38 | 1451.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 1458.60 | 1447.38 | 1451.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1446.90 | 1447.29 | 1451.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 1447.40 | 1447.29 | 1451.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1456.80 | 1439.32 | 1444.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 1451.00 | 1439.32 | 1444.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1457.70 | 1442.99 | 1445.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 1455.00 | 1442.99 | 1445.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 1454.10 | 1447.82 | 1447.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 1481.60 | 1456.19 | 1451.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 1472.70 | 1475.20 | 1465.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 15:00:00 | 1472.70 | 1475.20 | 1465.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1473.00 | 1481.74 | 1474.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 1476.00 | 1481.74 | 1474.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1470.10 | 1479.41 | 1473.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1470.10 | 1479.41 | 1473.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1467.30 | 1476.99 | 1473.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1482.70 | 1476.99 | 1473.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1463.00 | 1474.56 | 1472.85 | SL hit (close<static) qty=1.00 sl=1467.30 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 11:15:00 | 1454.80 | 1470.61 | 1471.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 12:15:00 | 1453.10 | 1467.11 | 1469.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 1441.40 | 1439.54 | 1450.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 1441.40 | 1439.54 | 1450.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1430.30 | 1437.75 | 1448.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:45:00 | 1422.40 | 1433.59 | 1444.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 1390.00 | 1375.26 | 1373.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 1390.00 | 1375.26 | 1373.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 1394.00 | 1379.01 | 1375.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 12:15:00 | 1407.40 | 1407.59 | 1398.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 13:00:00 | 1407.40 | 1407.59 | 1398.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1399.60 | 1405.99 | 1398.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:45:00 | 1398.70 | 1405.99 | 1398.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1400.10 | 1404.82 | 1398.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:00:00 | 1405.40 | 1403.70 | 1399.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1394.90 | 1401.94 | 1398.93 | SL hit (close<static) qty=1.00 sl=1396.70 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 1438.30 | 1440.49 | 1440.71 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 1442.90 | 1440.97 | 1440.91 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 1358.60 | 1425.16 | 1433.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 13:15:00 | 1339.00 | 1379.49 | 1406.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 1244.00 | 1234.75 | 1270.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:45:00 | 1249.10 | 1234.75 | 1270.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1248.00 | 1237.37 | 1244.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 1249.30 | 1237.37 | 1244.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1244.20 | 1238.74 | 1244.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:15:00 | 1233.40 | 1238.74 | 1244.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 14:15:00 | 1240.30 | 1225.46 | 1228.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 1242.30 | 1231.70 | 1230.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 1242.30 | 1231.70 | 1230.90 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1223.60 | 1230.08 | 1230.23 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 1237.00 | 1230.87 | 1230.47 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 1213.80 | 1227.35 | 1228.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 1206.60 | 1217.66 | 1222.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1222.20 | 1217.58 | 1221.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 1222.20 | 1217.58 | 1221.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1222.20 | 1217.58 | 1221.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1222.20 | 1217.58 | 1221.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1221.50 | 1218.37 | 1221.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1226.60 | 1218.37 | 1221.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1215.70 | 1217.83 | 1220.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 1210.80 | 1217.29 | 1220.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:00:00 | 1210.60 | 1215.28 | 1218.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:45:00 | 1208.90 | 1212.83 | 1215.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 1225.50 | 1218.75 | 1217.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1225.50 | 1218.75 | 1217.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 1227.30 | 1222.03 | 1219.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 12:15:00 | 1219.30 | 1221.49 | 1219.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 12:15:00 | 1219.30 | 1221.49 | 1219.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 1219.30 | 1221.49 | 1219.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 1219.30 | 1221.49 | 1219.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 1215.00 | 1220.19 | 1219.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:30:00 | 1215.00 | 1220.19 | 1219.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 1214.00 | 1218.95 | 1218.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 1214.00 | 1218.95 | 1218.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 1215.60 | 1218.28 | 1218.50 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1223.10 | 1219.24 | 1218.91 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 1215.60 | 1218.52 | 1218.61 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 1221.30 | 1219.07 | 1218.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 1223.30 | 1219.92 | 1219.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1240.20 | 1245.88 | 1239.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1240.20 | 1245.88 | 1239.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1240.20 | 1245.88 | 1239.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 1228.50 | 1245.88 | 1239.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1247.10 | 1246.13 | 1240.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:00:00 | 1250.00 | 1246.90 | 1241.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:45:00 | 1250.80 | 1247.38 | 1241.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 13:30:00 | 1249.20 | 1247.61 | 1242.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:15:00 | 1251.20 | 1247.61 | 1242.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1283.30 | 1293.35 | 1283.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1274.90 | 1293.35 | 1283.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1276.10 | 1289.90 | 1282.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 1276.10 | 1289.90 | 1282.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1274.30 | 1286.78 | 1281.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:45:00 | 1277.70 | 1286.78 | 1281.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 1260.50 | 1277.41 | 1278.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 1260.50 | 1277.41 | 1278.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1246.00 | 1264.94 | 1270.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 1202.00 | 1202.00 | 1211.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 13:45:00 | 1202.40 | 1202.00 | 1211.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1213.10 | 1205.64 | 1211.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1214.80 | 1205.64 | 1211.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1208.50 | 1206.21 | 1210.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 1209.30 | 1206.21 | 1210.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1199.20 | 1200.00 | 1204.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:45:00 | 1203.50 | 1200.00 | 1204.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1202.10 | 1200.23 | 1203.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 1210.30 | 1200.23 | 1203.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1215.00 | 1203.19 | 1204.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 1207.30 | 1204.19 | 1204.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 1200.40 | 1204.19 | 1204.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 13:15:00 | 1189.20 | 1176.34 | 1175.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 1189.20 | 1176.34 | 1175.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1205.00 | 1194.90 | 1187.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 1197.80 | 1197.87 | 1191.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 1197.80 | 1197.87 | 1191.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1200.80 | 1203.31 | 1200.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 1200.80 | 1203.31 | 1200.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1208.70 | 1204.39 | 1201.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:15:00 | 1212.90 | 1204.39 | 1201.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:15:00 | 1212.20 | 1216.00 | 1212.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 1203.10 | 1209.90 | 1210.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 1203.10 | 1209.90 | 1210.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1199.00 | 1205.93 | 1208.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 1153.70 | 1150.18 | 1160.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 11:00:00 | 1153.70 | 1150.18 | 1160.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1155.00 | 1151.29 | 1156.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1145.70 | 1151.29 | 1156.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1144.80 | 1149.99 | 1155.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 1142.00 | 1149.37 | 1154.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 1141.30 | 1147.44 | 1153.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 1140.50 | 1140.85 | 1147.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:45:00 | 1142.20 | 1142.02 | 1146.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1141.20 | 1141.86 | 1145.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 1141.20 | 1141.86 | 1145.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1125.20 | 1138.19 | 1143.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 1118.00 | 1133.97 | 1140.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:45:00 | 1117.00 | 1115.13 | 1121.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 13:15:00 | 1144.10 | 1122.89 | 1122.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 1144.10 | 1122.89 | 1122.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 15:15:00 | 1145.00 | 1130.61 | 1126.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 1126.40 | 1131.73 | 1129.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 14:15:00 | 1126.40 | 1131.73 | 1129.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 1126.40 | 1131.73 | 1129.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 1126.40 | 1131.73 | 1129.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1128.00 | 1130.99 | 1129.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1120.10 | 1130.67 | 1129.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1122.50 | 1129.04 | 1128.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 1124.90 | 1129.04 | 1128.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 1122.40 | 1127.71 | 1127.92 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 1146.20 | 1131.37 | 1129.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 1150.80 | 1137.84 | 1132.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1146.60 | 1147.96 | 1141.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:00:00 | 1146.60 | 1147.96 | 1141.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1147.40 | 1151.38 | 1146.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1147.40 | 1151.38 | 1146.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1140.30 | 1149.16 | 1146.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:30:00 | 1141.90 | 1149.16 | 1146.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1140.80 | 1147.49 | 1145.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:30:00 | 1138.10 | 1147.49 | 1145.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 1138.60 | 1143.96 | 1144.37 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 1173.20 | 1149.80 | 1146.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 1182.00 | 1156.24 | 1150.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 1185.70 | 1186.83 | 1177.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 12:00:00 | 1185.70 | 1186.83 | 1177.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1182.80 | 1185.17 | 1180.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:45:00 | 1177.70 | 1185.17 | 1180.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1189.20 | 1185.98 | 1181.01 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 1178.10 | 1183.06 | 1183.27 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 1191.50 | 1184.75 | 1184.02 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 1172.60 | 1182.75 | 1183.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 14:15:00 | 1167.40 | 1174.67 | 1177.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 15:15:00 | 1176.80 | 1175.10 | 1177.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 15:15:00 | 1176.80 | 1175.10 | 1177.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1176.80 | 1175.10 | 1177.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1177.00 | 1175.10 | 1177.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1178.20 | 1175.72 | 1177.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 1182.30 | 1175.72 | 1177.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1180.90 | 1176.76 | 1178.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:15:00 | 1185.30 | 1176.76 | 1178.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1185.00 | 1178.40 | 1178.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 1185.50 | 1178.40 | 1178.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 1184.50 | 1179.62 | 1179.30 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1173.30 | 1178.68 | 1179.01 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 1202.80 | 1179.86 | 1178.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1223.80 | 1198.66 | 1189.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 1226.10 | 1227.60 | 1216.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 14:00:00 | 1226.10 | 1227.60 | 1216.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1222.00 | 1224.80 | 1217.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:45:00 | 1232.80 | 1226.23 | 1220.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:00:00 | 1236.00 | 1228.18 | 1221.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:30:00 | 1234.20 | 1228.98 | 1222.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 1201.40 | 1223.18 | 1221.09 | SL hit (close<static) qty=1.00 sl=1213.90 alert=retest2 |

### Cycle 106 — SELL (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 12:15:00 | 1211.80 | 1219.16 | 1220.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 14:15:00 | 1206.90 | 1215.94 | 1218.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 12:15:00 | 1217.90 | 1211.03 | 1214.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 12:15:00 | 1217.90 | 1211.03 | 1214.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1217.90 | 1211.03 | 1214.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 1217.90 | 1211.03 | 1214.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1219.00 | 1212.62 | 1214.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:45:00 | 1217.50 | 1212.62 | 1214.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1231.30 | 1218.27 | 1217.05 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 1215.00 | 1220.11 | 1220.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 1205.60 | 1217.20 | 1218.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1218.70 | 1213.06 | 1215.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1218.70 | 1213.06 | 1215.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1218.70 | 1213.06 | 1215.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:45:00 | 1219.30 | 1213.06 | 1215.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1219.70 | 1214.39 | 1215.57 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 1220.30 | 1217.02 | 1216.61 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 1205.90 | 1214.73 | 1215.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1191.90 | 1206.18 | 1211.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1203.20 | 1200.27 | 1206.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:00:00 | 1203.20 | 1200.27 | 1206.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 1202.60 | 1201.70 | 1205.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 1198.10 | 1201.16 | 1204.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:00:00 | 1197.20 | 1199.52 | 1203.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1184.50 | 1199.16 | 1201.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:15:00 | 1138.19 | 1148.18 | 1153.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:15:00 | 1137.34 | 1148.18 | 1153.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 15:15:00 | 1125.27 | 1137.74 | 1146.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 1112.90 | 1107.50 | 1116.83 | SL hit (close>ema200) qty=0.50 sl=1107.50 alert=retest2 |

### Cycle 111 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1085.80 | 1082.22 | 1081.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 1093.10 | 1087.30 | 1084.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1080.90 | 1086.18 | 1084.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1080.90 | 1086.18 | 1084.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1080.90 | 1086.18 | 1084.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 1079.20 | 1086.18 | 1084.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1087.30 | 1086.40 | 1084.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:15:00 | 1093.90 | 1086.40 | 1084.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 1078.50 | 1086.13 | 1086.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 1078.50 | 1086.13 | 1086.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 1076.80 | 1084.27 | 1085.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1072.00 | 1066.39 | 1072.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1072.00 | 1066.39 | 1072.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1072.00 | 1066.39 | 1072.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1072.00 | 1066.39 | 1072.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1074.60 | 1068.03 | 1072.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 1075.60 | 1068.03 | 1072.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1070.50 | 1068.52 | 1072.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 1068.00 | 1068.52 | 1072.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 1079.50 | 1070.65 | 1072.42 | SL hit (close>static) qty=1.00 sl=1074.60 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 1081.60 | 1074.89 | 1074.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1090.60 | 1078.03 | 1075.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1087.90 | 1090.85 | 1085.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1087.90 | 1090.85 | 1085.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1087.90 | 1090.85 | 1085.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1084.70 | 1090.85 | 1085.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1084.70 | 1089.62 | 1085.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1088.70 | 1085.86 | 1085.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1081.40 | 1084.97 | 1084.82 | SL hit (close<static) qty=1.00 sl=1084.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 1080.50 | 1084.07 | 1084.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 1079.20 | 1082.60 | 1083.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 15:15:00 | 1082.20 | 1082.04 | 1083.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 15:15:00 | 1082.20 | 1082.04 | 1083.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1082.20 | 1082.04 | 1083.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1083.30 | 1082.04 | 1083.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1076.90 | 1081.01 | 1082.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 1081.20 | 1081.01 | 1082.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1080.50 | 1080.33 | 1081.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 1081.90 | 1080.33 | 1081.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 1075.00 | 1079.27 | 1081.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 14:00:00 | 1074.00 | 1078.21 | 1080.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:00:00 | 1072.00 | 1075.87 | 1078.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 11:15:00 | 1070.50 | 1065.60 | 1065.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 1070.50 | 1065.60 | 1065.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1080.00 | 1071.64 | 1068.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 1073.70 | 1074.27 | 1070.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 11:45:00 | 1073.10 | 1074.27 | 1070.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1110.60 | 1111.42 | 1100.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 1124.50 | 1110.39 | 1103.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 1119.20 | 1113.22 | 1106.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 1119.60 | 1111.87 | 1108.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 1099.70 | 1110.38 | 1108.46 | SL hit (close<static) qty=1.00 sl=1100.50 alert=retest2 |

### Cycle 116 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1092.80 | 1106.87 | 1107.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 1091.50 | 1102.21 | 1104.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1070.80 | 1068.72 | 1078.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 1070.00 | 1068.72 | 1078.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1077.30 | 1071.35 | 1077.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1079.00 | 1071.35 | 1077.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1074.00 | 1071.88 | 1077.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1068.00 | 1071.88 | 1077.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1069.00 | 1071.30 | 1076.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1058.60 | 1065.94 | 1071.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 1079.20 | 1068.57 | 1068.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 1079.20 | 1068.57 | 1068.26 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1057.00 | 1066.86 | 1068.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 1048.50 | 1063.19 | 1066.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 910.00 | 901.33 | 922.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 910.00 | 901.33 | 922.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 919.30 | 906.55 | 921.14 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 949.70 | 924.96 | 924.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 954.70 | 937.18 | 930.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 955.35 | 961.46 | 951.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 955.35 | 961.46 | 951.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 955.35 | 961.46 | 951.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 952.85 | 961.46 | 951.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 951.75 | 959.76 | 952.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 951.75 | 959.76 | 952.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 961.50 | 960.11 | 952.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 947.50 | 960.11 | 952.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 950.35 | 958.16 | 952.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 950.35 | 958.16 | 952.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 915.00 | 949.53 | 949.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 915.00 | 949.53 | 949.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 921.55 | 943.93 | 946.75 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 961.85 | 949.59 | 948.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 963.50 | 952.37 | 949.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1040.75 | 1041.46 | 1019.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1040.75 | 1041.46 | 1019.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1029.25 | 1041.39 | 1031.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 1029.25 | 1041.39 | 1031.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 1042.15 | 1041.54 | 1032.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:45:00 | 1046.35 | 1043.24 | 1035.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1050.05 | 1079.10 | 1081.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1050.05 | 1079.10 | 1081.16 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1089.10 | 1078.65 | 1078.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 1096.00 | 1084.89 | 1081.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 1092.55 | 1093.15 | 1088.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:00:00 | 1092.55 | 1093.15 | 1088.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1088.85 | 1093.04 | 1089.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 1088.85 | 1093.04 | 1089.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1093.00 | 1093.03 | 1089.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:45:00 | 1096.60 | 1093.99 | 1090.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 1086.50 | 1096.90 | 1094.25 | SL hit (close<static) qty=1.00 sl=1089.15 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 1072.45 | 1092.01 | 1092.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 1068.50 | 1087.31 | 1090.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 1078.00 | 1075.97 | 1081.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 1078.00 | 1075.97 | 1081.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 1078.00 | 1075.97 | 1081.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:30:00 | 1081.30 | 1075.97 | 1081.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1076.55 | 1076.09 | 1081.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 1080.15 | 1076.09 | 1081.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1080.70 | 1077.01 | 1081.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 1081.45 | 1077.01 | 1081.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1071.00 | 1075.81 | 1080.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:15:00 | 1068.10 | 1075.81 | 1080.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:30:00 | 1066.90 | 1072.27 | 1077.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1067.15 | 1071.98 | 1075.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:15:00 | 1014.69 | 1025.90 | 1040.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:15:00 | 1013.56 | 1025.90 | 1040.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:15:00 | 1013.79 | 1025.90 | 1040.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 961.29 | 992.28 | 1007.96 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 860.60 | 854.46 | 854.26 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 824.80 | 848.38 | 851.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 817.00 | 830.79 | 840.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 752.80 | 735.92 | 754.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 752.80 | 735.92 | 754.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 752.80 | 735.92 | 754.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 750.15 | 735.92 | 754.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 744.90 | 737.72 | 753.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 741.90 | 738.94 | 752.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 740.20 | 738.94 | 752.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:15:00 | 735.10 | 740.11 | 750.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:15:00 | 704.80 | 726.95 | 740.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:15:00 | 703.19 | 722.43 | 737.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 14:15:00 | 698.35 | 711.15 | 728.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 690.00 | 685.78 | 701.64 | SL hit (close>ema200) qty=0.50 sl=685.78 alert=retest2 |

### Cycle 127 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 693.50 | 690.46 | 690.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 701.70 | 692.71 | 691.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 816.00 | 818.83 | 797.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 816.00 | 818.83 | 797.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 816.00 | 818.83 | 797.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 824.80 | 820.02 | 799.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 842.00 | 821.77 | 808.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 864.25 | 872.74 | 873.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 864.25 | 872.74 | 873.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 855.70 | 867.81 | 870.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 853.50 | 847.98 | 855.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 10:45:00 | 852.55 | 847.98 | 855.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 854.25 | 849.23 | 855.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:15:00 | 862.40 | 849.23 | 855.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 865.00 | 852.38 | 856.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 864.00 | 852.38 | 856.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 868.45 | 855.60 | 857.18 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 880.00 | 860.48 | 859.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 885.30 | 871.08 | 864.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 901.40 | 904.56 | 893.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 15:15:00 | 893.55 | 900.56 | 895.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 893.55 | 900.56 | 895.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 906.30 | 900.56 | 895.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:15:00 | 1099.95 | 2024-05-13 15:15:00 | 1140.00 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2024-06-04 14:00:00 | 1361.70 | 2024-06-04 14:15:00 | 1313.90 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2024-06-06 11:15:00 | 1386.00 | 2024-06-06 11:15:00 | 1375.55 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2024-07-10 13:00:00 | 1532.85 | 2024-07-10 13:15:00 | 1522.65 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-07-22 09:15:00 | 1422.85 | 2024-07-23 15:15:00 | 1361.16 | PARTIAL | 0.50 | 4.34% |
| SELL | retest2 | 2024-07-22 11:00:00 | 1427.65 | 2024-07-23 15:15:00 | 1361.16 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2024-07-22 09:15:00 | 1422.85 | 2024-07-24 14:15:00 | 1404.30 | STOP_HIT | 0.50 | 1.30% |
| SELL | retest2 | 2024-07-22 11:00:00 | 1427.65 | 2024-07-24 14:15:00 | 1404.30 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2024-07-22 12:15:00 | 1432.80 | 2024-07-29 15:15:00 | 1356.27 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2024-07-22 13:00:00 | 1432.80 | 2024-07-30 09:15:00 | 1351.71 | PARTIAL | 0.50 | 5.66% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1401.05 | 2024-07-30 14:15:00 | 1331.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 12:15:00 | 1432.80 | 2024-08-01 09:15:00 | 1280.57 | TARGET_HIT | 0.50 | 10.62% |
| SELL | retest2 | 2024-07-22 13:00:00 | 1432.80 | 2024-08-01 09:15:00 | 1284.89 | TARGET_HIT | 0.50 | 10.32% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1401.05 | 2024-08-01 12:15:00 | 1260.94 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-16 09:30:00 | 1310.00 | 2024-08-19 10:15:00 | 1283.40 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-08-16 13:45:00 | 1307.90 | 2024-08-19 10:15:00 | 1283.40 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest1 | 2024-09-05 09:15:00 | 1208.75 | 2024-09-11 12:15:00 | 1202.25 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest1 | 2024-09-05 11:00:00 | 1207.45 | 2024-09-11 12:15:00 | 1202.25 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest1 | 2024-09-05 12:15:00 | 1207.85 | 2024-09-11 12:15:00 | 1202.25 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest1 | 2024-09-05 13:00:00 | 1207.55 | 2024-09-11 12:15:00 | 1202.25 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-09-09 12:30:00 | 1183.35 | 2024-09-11 15:15:00 | 1204.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-09-10 15:00:00 | 1185.70 | 2024-09-11 15:15:00 | 1204.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-09-11 09:15:00 | 1183.00 | 2024-09-11 15:15:00 | 1204.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-09-11 11:30:00 | 1183.60 | 2024-09-11 15:15:00 | 1204.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-09-19 14:45:00 | 1291.30 | 2024-09-23 09:15:00 | 1420.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-10-04 13:30:00 | 1163.55 | 2024-10-07 09:15:00 | 1206.50 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest1 | 2024-10-04 14:00:00 | 1162.15 | 2024-10-07 09:15:00 | 1206.50 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2024-10-07 15:00:00 | 1181.75 | 2024-10-09 10:15:00 | 1222.95 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2024-10-21 13:00:00 | 1149.00 | 2024-10-22 13:15:00 | 1091.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 13:30:00 | 1147.75 | 2024-10-22 13:15:00 | 1090.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 13:00:00 | 1149.00 | 2024-10-23 14:15:00 | 1087.10 | STOP_HIT | 0.50 | 5.39% |
| SELL | retest2 | 2024-10-21 13:30:00 | 1147.75 | 2024-10-23 14:15:00 | 1087.10 | STOP_HIT | 0.50 | 5.28% |
| SELL | retest2 | 2024-11-11 15:00:00 | 1187.05 | 2024-11-12 09:15:00 | 1277.25 | STOP_HIT | 1.00 | -7.60% |
| BUY | retest2 | 2024-11-19 09:15:00 | 1283.50 | 2024-11-25 14:15:00 | 1219.45 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest2 | 2024-11-22 09:15:00 | 1280.00 | 2024-11-25 14:15:00 | 1219.45 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2024-11-22 15:15:00 | 1249.50 | 2024-11-25 14:15:00 | 1219.45 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-11-29 09:30:00 | 1298.60 | 2024-11-29 13:15:00 | 1272.85 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-12-03 15:00:00 | 1295.90 | 2024-12-12 10:15:00 | 1425.49 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-23 12:45:00 | 1435.90 | 2024-12-31 09:15:00 | 1364.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-23 12:45:00 | 1435.90 | 2024-12-31 13:15:00 | 1387.95 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2024-12-24 09:15:00 | 1411.05 | 2025-01-03 12:15:00 | 1392.80 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2025-01-09 09:30:00 | 1324.00 | 2025-01-13 09:15:00 | 1257.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:45:00 | 1325.40 | 2025-01-13 09:15:00 | 1259.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:15:00 | 1324.00 | 2025-01-13 09:15:00 | 1257.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:30:00 | 1324.00 | 2025-01-13 12:15:00 | 1191.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 11:45:00 | 1325.40 | 2025-01-13 12:15:00 | 1192.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 12:15:00 | 1324.00 | 2025-01-13 12:15:00 | 1191.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 1091.55 | 2025-01-27 11:15:00 | 1119.95 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-01-24 14:30:00 | 1096.45 | 2025-01-27 11:15:00 | 1119.95 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-01-30 09:15:00 | 1152.05 | 2025-02-01 14:15:00 | 1267.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-10 11:00:00 | 1183.75 | 2025-02-17 09:15:00 | 1124.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 11:00:00 | 1183.70 | 2025-02-17 09:15:00 | 1124.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 11:00:00 | 1183.75 | 2025-02-17 11:15:00 | 1159.35 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2025-02-12 11:00:00 | 1183.70 | 2025-02-17 11:15:00 | 1159.35 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2025-02-13 10:45:00 | 1179.25 | 2025-02-18 12:15:00 | 1170.50 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-02-25 09:45:00 | 1213.30 | 2025-02-27 11:15:00 | 1191.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-02-27 09:15:00 | 1211.35 | 2025-02-27 11:15:00 | 1191.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-02-27 11:00:00 | 1201.25 | 2025-02-27 11:15:00 | 1191.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-03-18 11:45:00 | 1064.80 | 2025-03-18 13:15:00 | 1101.90 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-03-18 12:30:00 | 1067.95 | 2025-03-18 13:15:00 | 1101.90 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2025-03-21 15:15:00 | 1188.00 | 2025-03-28 15:15:00 | 1196.90 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2025-04-25 09:15:00 | 1382.50 | 2025-04-25 13:15:00 | 1300.30 | STOP_HIT | 1.00 | -5.95% |
| BUY | retest2 | 2025-04-25 10:00:00 | 1334.10 | 2025-04-25 13:15:00 | 1300.30 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-04-29 09:15:00 | 1304.70 | 2025-04-30 09:15:00 | 1353.60 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-05-08 14:45:00 | 1293.70 | 2025-05-09 09:15:00 | 1229.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 14:45:00 | 1293.70 | 2025-05-12 09:15:00 | 1292.30 | STOP_HIT | 0.50 | 0.11% |
| SELL | retest2 | 2025-05-12 09:30:00 | 1284.40 | 2025-05-12 13:15:00 | 1306.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-05-12 11:15:00 | 1294.50 | 2025-05-12 13:15:00 | 1306.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-05-21 09:15:00 | 1404.10 | 2025-05-30 09:15:00 | 1437.80 | STOP_HIT | 1.00 | 2.40% |
| SELL | retest2 | 2025-06-02 15:15:00 | 1430.20 | 2025-06-03 09:15:00 | 1457.40 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-06-04 09:15:00 | 1452.60 | 2025-06-10 13:15:00 | 1469.70 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-06-04 10:00:00 | 1451.20 | 2025-06-10 13:15:00 | 1469.70 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2025-06-04 11:45:00 | 1453.80 | 2025-06-10 13:15:00 | 1469.70 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2025-06-05 09:15:00 | 1463.00 | 2025-06-10 13:15:00 | 1469.70 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-06-06 09:15:00 | 1471.00 | 2025-06-10 13:15:00 | 1469.70 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-06-12 09:15:00 | 1466.00 | 2025-06-16 14:15:00 | 1470.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1482.70 | 2025-06-25 10:15:00 | 1463.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-06-27 11:45:00 | 1422.40 | 2025-07-09 10:15:00 | 1390.00 | STOP_HIT | 1.00 | 2.28% |
| BUY | retest2 | 2025-07-14 10:00:00 | 1405.40 | 2025-07-14 10:15:00 | 1394.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-14 12:30:00 | 1404.00 | 2025-07-22 12:15:00 | 1438.30 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2025-07-14 13:15:00 | 1406.80 | 2025-07-22 12:15:00 | 1438.30 | STOP_HIT | 1.00 | 2.24% |
| SELL | retest2 | 2025-07-31 14:15:00 | 1233.40 | 2025-08-04 15:15:00 | 1242.30 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-08-04 14:15:00 | 1240.30 | 2025-08-04 15:15:00 | 1242.30 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-08 10:30:00 | 1210.80 | 2025-08-11 15:15:00 | 1225.50 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-08-08 14:00:00 | 1210.60 | 2025-08-11 15:15:00 | 1225.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-08-11 11:45:00 | 1208.90 | 2025-08-11 15:15:00 | 1225.50 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-08-19 12:00:00 | 1250.00 | 2025-08-22 14:15:00 | 1260.50 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-08-19 12:45:00 | 1250.80 | 2025-08-22 14:15:00 | 1260.50 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-08-19 13:30:00 | 1249.20 | 2025-08-22 14:15:00 | 1260.50 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2025-08-19 14:15:00 | 1251.20 | 2025-08-22 14:15:00 | 1260.50 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-09-04 10:30:00 | 1207.30 | 2025-09-15 13:15:00 | 1189.20 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-09-04 11:15:00 | 1200.40 | 2025-09-15 13:15:00 | 1189.20 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-09-19 13:15:00 | 1212.90 | 2025-09-23 12:15:00 | 1203.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-23 10:15:00 | 1212.20 | 2025-09-23 12:15:00 | 1203.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-09-30 10:30:00 | 1142.00 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-09-30 11:30:00 | 1141.30 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-10-01 10:00:00 | 1140.50 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-10-01 12:45:00 | 1142.20 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-10-03 10:30:00 | 1118.00 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-10-07 09:45:00 | 1117.00 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-11-06 12:45:00 | 1232.80 | 2025-11-07 09:15:00 | 1201.40 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-11-06 14:00:00 | 1236.00 | 2025-11-07 09:15:00 | 1201.40 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-11-06 14:30:00 | 1234.20 | 2025-11-07 09:15:00 | 1201.40 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-11-20 09:15:00 | 1198.10 | 2025-12-01 11:15:00 | 1138.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 11:00:00 | 1197.20 | 2025-12-01 11:15:00 | 1137.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1184.50 | 2025-12-01 15:15:00 | 1125.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 09:15:00 | 1198.10 | 2025-12-04 11:15:00 | 1112.90 | STOP_HIT | 0.50 | 7.11% |
| SELL | retest2 | 2025-11-20 11:00:00 | 1197.20 | 2025-12-04 11:15:00 | 1112.90 | STOP_HIT | 0.50 | 7.04% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1184.50 | 2025-12-04 11:15:00 | 1112.90 | STOP_HIT | 0.50 | 6.04% |
| BUY | retest2 | 2025-12-15 11:15:00 | 1093.90 | 2025-12-16 12:15:00 | 1078.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-18 14:15:00 | 1068.00 | 2025-12-19 09:15:00 | 1079.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1088.70 | 2025-12-24 09:15:00 | 1081.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-26 14:00:00 | 1074.00 | 2026-01-01 11:15:00 | 1070.50 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-12-29 10:00:00 | 1072.00 | 2026-01-01 11:15:00 | 1070.50 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2026-01-07 09:15:00 | 1124.50 | 2026-01-08 11:15:00 | 1099.70 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-01-07 10:45:00 | 1119.20 | 2026-01-08 11:15:00 | 1099.70 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-01-08 09:15:00 | 1119.60 | 2026-01-08 11:15:00 | 1099.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1058.60 | 2026-01-16 11:15:00 | 1079.20 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-02-06 13:45:00 | 1046.35 | 2026-02-13 09:15:00 | 1050.05 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2026-02-18 11:45:00 | 1096.60 | 2026-02-19 10:15:00 | 1086.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-02-20 15:15:00 | 1068.10 | 2026-02-26 10:15:00 | 1014.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:30:00 | 1066.90 | 2026-02-26 10:15:00 | 1013.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1067.15 | 2026-02-26 10:15:00 | 1013.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 15:15:00 | 1068.10 | 2026-03-02 09:15:00 | 961.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 11:30:00 | 1066.90 | 2026-03-02 09:15:00 | 960.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1067.15 | 2026-03-02 09:15:00 | 960.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 741.90 | 2026-03-27 10:15:00 | 704.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 12:15:00 | 740.20 | 2026-03-27 11:15:00 | 703.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:15:00 | 735.10 | 2026-03-27 14:15:00 | 698.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 741.90 | 2026-04-01 09:15:00 | 690.00 | STOP_HIT | 0.50 | 7.00% |
| SELL | retest2 | 2026-03-25 12:15:00 | 740.20 | 2026-04-01 09:15:00 | 690.00 | STOP_HIT | 0.50 | 6.78% |
| SELL | retest2 | 2026-03-25 14:15:00 | 735.10 | 2026-04-01 09:15:00 | 690.00 | STOP_HIT | 0.50 | 6.14% |
| BUY | retest2 | 2026-04-13 11:00:00 | 824.80 | 2026-04-23 12:15:00 | 864.25 | STOP_HIT | 1.00 | 4.78% |
| BUY | retest2 | 2026-04-15 09:15:00 | 842.00 | 2026-04-23 12:15:00 | 864.25 | STOP_HIT | 1.00 | 2.64% |
