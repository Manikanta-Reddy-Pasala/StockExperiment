# Pidilite Industries Ltd. (PIDILITIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1472.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 1 |
| ALERT3 | 78 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 50 |
| PARTIAL | 7 |
| TARGET_HIT | 8 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 38
- **Target hits / Stop hits / Partials:** 8 / 43 / 7
- **Avg / median % per leg:** 1.26% / -0.72%
- **Sum % (uncompounded):** 72.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 6 | 40.0% | 6 | 9 | 0 | 2.74% | 41.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 6 | 40.0% | 6 | 9 | 0 | 2.74% | 41.1% |
| SELL (all) | 43 | 14 | 32.6% | 2 | 34 | 7 | 0.74% | 31.8% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.53% | 7.1% |
| SELL @ 3rd Alert (retest2) | 41 | 12 | 29.3% | 2 | 33 | 6 | 0.60% | 24.7% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.53% | 7.1% |
| retest2 (combined) | 56 | 18 | 32.1% | 8 | 42 | 6 | 1.18% | 65.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 14:15:00 | 1253.78 | 1285.96 | 1285.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-01 09:15:00 | 1251.97 | 1281.04 | 1283.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 15:15:00 | 1260.00 | 1259.81 | 1268.92 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 09:15:00 | 1252.50 | 1259.81 | 1268.92 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 11:15:00 | 1189.88 | 1233.51 | 1248.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-10-31 10:15:00 | 1226.78 | 1209.12 | 1231.53 | SL hit (close>ema200) qty=0.50 sl=1209.12 alert=retest1 |

### Cycle 2 — BUY (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 09:15:00 | 1279.45 | 1235.89 | 1235.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 12:15:00 | 1291.97 | 1240.15 | 1238.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 10:15:00 | 1340.50 | 1341.07 | 1310.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-19 11:00:00 | 1340.50 | 1341.07 | 1310.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1295.00 | 1340.87 | 1312.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 1295.00 | 1340.87 | 1312.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 1296.88 | 1340.43 | 1312.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 09:15:00 | 1312.48 | 1336.96 | 1311.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 11:15:00 | 1283.90 | 1330.15 | 1309.96 | SL hit (close<static) qty=1.00 sl=1286.18 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 11:15:00 | 1546.60 | 1585.16 | 1585.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 1539.10 | 1578.73 | 1581.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 1547.68 | 1539.37 | 1556.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 09:45:00 | 1547.98 | 1539.37 | 1556.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 1557.60 | 1539.71 | 1556.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:00:00 | 1557.60 | 1539.71 | 1556.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 1567.70 | 1539.99 | 1556.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:00:00 | 1567.70 | 1539.99 | 1556.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1567.00 | 1540.25 | 1556.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:30:00 | 1567.53 | 1540.25 | 1556.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1570.00 | 1541.30 | 1557.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:30:00 | 1568.40 | 1541.30 | 1557.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1570.20 | 1552.25 | 1560.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:00:00 | 1570.20 | 1552.25 | 1560.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1574.20 | 1552.47 | 1561.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:00:00 | 1574.20 | 1552.47 | 1561.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1488.15 | 1452.62 | 1490.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1488.15 | 1452.62 | 1490.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1488.00 | 1452.97 | 1490.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 1488.00 | 1452.97 | 1490.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1472.05 | 1453.16 | 1490.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 1489.53 | 1453.16 | 1490.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1483.03 | 1447.71 | 1479.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1483.03 | 1447.71 | 1479.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1477.68 | 1448.01 | 1479.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:30:00 | 1493.70 | 1448.01 | 1479.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1475.88 | 1448.78 | 1479.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:00:00 | 1475.88 | 1448.78 | 1479.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 1481.08 | 1449.11 | 1479.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:00:00 | 1481.08 | 1449.11 | 1479.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 1471.10 | 1449.32 | 1479.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:45:00 | 1471.18 | 1449.32 | 1479.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 1480.78 | 1450.05 | 1479.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 15:00:00 | 1480.78 | 1450.05 | 1479.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 1478.53 | 1450.33 | 1479.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 1464.53 | 1450.33 | 1479.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1460.00 | 1450.43 | 1479.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 1451.73 | 1451.36 | 1478.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 09:45:00 | 1453.45 | 1451.40 | 1478.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 11:30:00 | 1458.93 | 1451.68 | 1478.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 13:45:00 | 1457.73 | 1451.82 | 1478.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1379.14 | 1439.40 | 1465.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1380.78 | 1439.40 | 1465.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1385.98 | 1439.40 | 1465.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1384.84 | 1439.40 | 1465.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 10:15:00 | 1313.04 | 1404.91 | 1438.96 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 14:15:00 | 1513.15 | 1425.94 | 1425.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 1521.30 | 1427.73 | 1426.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 10:15:00 | 1473.05 | 1476.78 | 1457.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 11:00:00 | 1473.05 | 1476.78 | 1457.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1493.75 | 1520.24 | 1499.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 1496.35 | 1520.24 | 1499.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:30:00 | 1500.85 | 1519.27 | 1499.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 1498.30 | 1508.25 | 1497.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 14:00:00 | 1501.75 | 1508.17 | 1497.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 1505.25 | 1508.14 | 1497.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:45:00 | 1500.55 | 1508.14 | 1497.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1499.45 | 1508.06 | 1497.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 1509.10 | 1508.03 | 1498.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 1506.25 | 1507.92 | 1498.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 1506.00 | 1519.94 | 1508.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 1490.00 | 1519.09 | 1508.21 | SL hit (close<static) qty=1.00 sl=1496.05 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 1452.20 | 1500.28 | 1500.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1442.70 | 1498.79 | 1499.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1482.95 | 1478.34 | 1488.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:45:00 | 1483.50 | 1478.34 | 1488.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1482.05 | 1478.31 | 1487.94 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1540.65 | 1495.59 | 1495.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 1564.35 | 1514.12 | 1505.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 09:15:00 | 1513.90 | 1520.05 | 1509.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:00:00 | 1513.90 | 1520.05 | 1509.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1526.70 | 1536.07 | 1524.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 1524.05 | 1536.07 | 1524.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1524.95 | 1535.78 | 1524.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 1524.95 | 1535.78 | 1524.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1524.10 | 1535.67 | 1524.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:45:00 | 1522.30 | 1535.67 | 1524.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1522.40 | 1535.54 | 1524.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 1522.40 | 1535.54 | 1524.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1521.55 | 1535.40 | 1524.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1521.55 | 1535.40 | 1524.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1529.95 | 1535.24 | 1524.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:30:00 | 1531.20 | 1535.14 | 1524.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 1519.35 | 1534.82 | 1524.53 | SL hit (close<static) qty=1.00 sl=1522.75 alert=retest2 |

### Cycle 7 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1451.70 | 1516.54 | 1516.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 1445.00 | 1505.99 | 1509.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 1489.30 | 1487.56 | 1498.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 10:45:00 | 1491.30 | 1487.56 | 1498.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1491.70 | 1485.53 | 1495.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:30:00 | 1495.70 | 1485.53 | 1495.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1492.10 | 1485.72 | 1495.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 1484.50 | 1485.68 | 1495.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:30:00 | 1488.40 | 1485.49 | 1495.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1475.00 | 1485.73 | 1495.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 1487.20 | 1482.47 | 1492.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1484.30 | 1474.66 | 1483.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 1486.40 | 1474.66 | 1483.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1486.10 | 1474.77 | 1483.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1486.10 | 1474.77 | 1483.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1482.50 | 1474.85 | 1483.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:30:00 | 1477.70 | 1474.85 | 1483.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1476.80 | 1465.25 | 1475.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:30:00 | 1478.40 | 1466.41 | 1475.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:30:00 | 1477.60 | 1466.53 | 1475.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1477.40 | 1466.64 | 1475.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 1475.90 | 1466.64 | 1475.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | SL hit (close>static) qty=1.00 sl=1497.20 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 1497.70 | 1471.73 | 1471.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 10:15:00 | 1508.30 | 1472.60 | 1472.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1464.50 | 1476.06 | 1473.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 1464.50 | 1476.06 | 1473.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1464.50 | 1476.06 | 1473.91 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 10:15:00 | 1426.50 | 1471.86 | 1471.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1374.20 | 1467.48 | 1469.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1371.00 | 1364.17 | 1403.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:45:00 | 1372.00 | 1364.17 | 1403.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 1390.00 | 1356.65 | 1391.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 12:00:00 | 1390.00 | 1356.65 | 1391.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 12:15:00 | 1391.10 | 1356.99 | 1391.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 13:00:00 | 1391.10 | 1356.99 | 1391.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 1392.60 | 1357.35 | 1391.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:00:00 | 1392.60 | 1357.35 | 1391.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 1394.30 | 1357.72 | 1391.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:30:00 | 1394.90 | 1357.72 | 1391.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 1398.00 | 1358.12 | 1391.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 1391.00 | 1358.12 | 1391.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 1396.70 | 1358.85 | 1391.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:30:00 | 1397.30 | 1358.85 | 1391.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 1391.80 | 1359.50 | 1391.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:15:00 | 1396.20 | 1359.50 | 1391.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 1393.00 | 1359.84 | 1391.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 15:15:00 | 1389.00 | 1360.16 | 1391.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 09:15:00 | 1409.80 | 1360.94 | 1391.74 | SL hit (close>static) qty=1.00 sl=1396.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-09-25 09:15:00 | 1252.50 | 2023-10-19 11:15:00 | 1189.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2023-09-25 09:15:00 | 1252.50 | 2023-10-31 10:15:00 | 1226.78 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2023-11-01 09:45:00 | 1220.80 | 2023-11-08 11:15:00 | 1236.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2023-11-06 09:15:00 | 1214.03 | 2023-11-08 11:15:00 | 1236.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2023-11-06 12:45:00 | 1223.00 | 2023-11-08 11:15:00 | 1236.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2023-11-08 09:45:00 | 1222.85 | 2023-11-08 11:15:00 | 1236.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-11-09 11:30:00 | 1229.58 | 2023-11-17 09:15:00 | 1261.55 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2023-11-16 09:15:00 | 1223.60 | 2023-11-17 09:15:00 | 1261.55 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2023-11-16 14:00:00 | 1230.50 | 2023-11-17 09:15:00 | 1261.55 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2023-11-16 15:00:00 | 1229.28 | 2023-11-17 09:15:00 | 1261.55 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-01-24 09:15:00 | 1312.48 | 2024-01-29 11:15:00 | 1283.90 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-02-06 13:00:00 | 1307.45 | 2024-03-07 09:15:00 | 1438.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-06 15:00:00 | 1307.53 | 2024-03-07 09:15:00 | 1438.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-08 15:00:00 | 1312.35 | 2024-03-07 09:15:00 | 1443.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-09 12:15:00 | 1324.43 | 2024-03-07 09:15:00 | 1454.53 | TARGET_HIT | 1.00 | 9.82% |
| BUY | retest2 | 2024-02-09 13:45:00 | 1322.30 | 2024-03-07 10:15:00 | 1456.87 | TARGET_HIT | 1.00 | 10.18% |
| BUY | retest2 | 2024-02-09 14:15:00 | 1325.03 | 2024-03-07 10:15:00 | 1457.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-05 09:15:00 | 1451.73 | 2025-02-17 09:15:00 | 1379.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 09:45:00 | 1453.45 | 2025-02-17 09:15:00 | 1380.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 11:30:00 | 1458.93 | 2025-02-17 09:15:00 | 1385.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 13:45:00 | 1457.73 | 2025-02-17 09:15:00 | 1384.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 09:15:00 | 1451.73 | 2025-03-03 10:15:00 | 1313.04 | TARGET_HIT | 0.50 | 9.55% |
| SELL | retest2 | 2025-02-05 09:45:00 | 1453.45 | 2025-03-03 10:15:00 | 1311.96 | TARGET_HIT | 0.50 | 9.73% |
| SELL | retest2 | 2025-02-05 11:30:00 | 1458.93 | 2025-03-20 11:15:00 | 1382.95 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2025-02-05 13:45:00 | 1457.73 | 2025-03-20 11:15:00 | 1382.95 | STOP_HIT | 0.50 | 5.13% |
| SELL | retest2 | 2025-03-21 13:15:00 | 1408.75 | 2025-03-21 15:15:00 | 1417.48 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-03-21 14:45:00 | 1408.23 | 2025-03-21 15:15:00 | 1417.48 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-03-24 09:30:00 | 1404.63 | 2025-03-25 09:15:00 | 1423.33 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-03-24 14:15:00 | 1409.50 | 2025-03-25 09:15:00 | 1423.33 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-06-13 10:15:00 | 1496.35 | 2025-07-14 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-06-13 13:30:00 | 1500.85 | 2025-07-14 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1498.30 | 2025-07-14 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-06-25 14:00:00 | 1501.75 | 2025-07-24 13:15:00 | 1452.20 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2025-06-26 09:30:00 | 1509.10 | 2025-07-24 13:15:00 | 1452.20 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-06-26 13:00:00 | 1506.25 | 2025-07-24 13:15:00 | 1452.20 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-07-11 12:30:00 | 1506.00 | 2025-07-24 13:15:00 | 1452.20 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-09-22 10:30:00 | 1531.20 | 2025-09-22 14:15:00 | 1519.35 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-19 10:30:00 | 1484.50 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-11-20 11:30:00 | 1488.40 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1475.00 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-11-26 13:15:00 | 1487.20 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-12-16 12:30:00 | 1477.70 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1476.80 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-01-02 13:30:00 | 1478.40 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-01-02 14:30:00 | 1477.60 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1486.40 | 2026-01-13 14:15:00 | 1498.60 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-14 12:45:00 | 1487.70 | 2026-01-21 10:15:00 | 1413.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:00:00 | 1484.10 | 2026-02-02 09:15:00 | 1409.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:45:00 | 1487.70 | 2026-02-04 09:15:00 | 1469.50 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2026-01-16 11:00:00 | 1484.10 | 2026-02-04 09:15:00 | 1469.50 | STOP_HIT | 0.50 | 0.98% |
| SELL | retest2 | 2026-02-06 14:30:00 | 1488.60 | 2026-02-12 14:15:00 | 1498.20 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-19 14:15:00 | 1461.40 | 2026-02-23 09:15:00 | 1475.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-02-20 09:15:00 | 1462.80 | 2026-02-23 09:15:00 | 1475.20 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-20 09:45:00 | 1462.00 | 2026-02-23 09:15:00 | 1475.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-02-20 15:15:00 | 1463.00 | 2026-02-23 09:15:00 | 1475.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-04-20 15:15:00 | 1389.00 | 2026-04-21 09:15:00 | 1409.80 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-04-24 12:15:00 | 1390.60 | 2026-04-24 15:15:00 | 1398.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1390.00 | 2026-05-06 10:15:00 | 1409.00 | STOP_HIT | 1.00 | -1.37% |
