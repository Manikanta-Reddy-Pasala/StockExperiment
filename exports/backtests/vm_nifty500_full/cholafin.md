# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1562.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 5 |
| ENTRY1 | 10 |
| ENTRY2 | 2 |
| EXIT | 10 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / EMA400 exits:** 4 / 8
- **Total realized P&L (per unit):** 183.17
- **Avg P&L per closed trade:** 15.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 12:15:00 | 1122.70 | 1092.24 | 1092.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 13:15:00 | 1126.00 | 1092.57 | 1092.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 09:15:00 | 1198.55 | 1203.21 | 1168.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-03 09:15:00 | 1215.50 | 1180.53 | 1165.96 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-11-03 11:15:00 | 1146.70 | 1180.22 | 1165.95 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 1122.65 | 1157.16 | 1157.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 13:15:00 | 1118.85 | 1156.45 | 1156.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 09:15:00 | 1143.95 | 1140.74 | 1147.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-05 09:15:00 | 1137.25 | 1141.16 | 1147.93 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 1137.25 | 1141.16 | 1147.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-12-06 09:15:00 | 1148.15 | 1141.13 | 1147.68 | Close above EMA400 |

### Cycle 3 — BUY (started 2023-12-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 13:15:00 | 1228.90 | 1152.64 | 1152.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 15:15:00 | 1230.00 | 1154.16 | 1153.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 11:15:00 | 1211.75 | 1217.96 | 1195.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-11 09:15:00 | 1230.35 | 1217.94 | 1195.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 15:15:00 | 1220.25 | 1239.42 | 1213.16 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-24 12:15:00 | 1225.75 | 1238.63 | 1213.28 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-01-30 14:15:00 | 1212.30 | 1238.01 | 1215.72 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 1114.55 | 1199.03 | 1199.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 1105.15 | 1197.26 | 1198.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 13:15:00 | 1091.75 | 1090.44 | 1123.30 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 09:15:00 | 1192.15 | 1143.23 | 1143.06 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-04-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 15:15:00 | 1130.00 | 1143.31 | 1143.32 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 09:15:00 | 1168.50 | 1143.43 | 1143.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 10:15:00 | 1174.50 | 1143.74 | 1143.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 1227.00 | 1239.07 | 1208.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-31 11:15:00 | 1245.25 | 1237.74 | 1209.09 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1219.20 | 1241.27 | 1212.85 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-05 10:15:00 | 1273.75 | 1241.44 | 1213.77 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-25 09:15:00 | 1349.50 | 1395.13 | 1349.98 | Close below EMA400 |

### Cycle 8 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 1270.50 | 1465.54 | 1465.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 1264.80 | 1432.42 | 1448.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 15:15:00 | 1298.30 | 1292.65 | 1345.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-05 10:15:00 | 1284.45 | 1292.55 | 1345.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 1339.00 | 1293.78 | 1339.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-11 09:15:00 | 1342.90 | 1294.27 | 1339.23 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 12:15:00 | 1332.15 | 1291.83 | 1291.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-10 14:15:00 | 1345.65 | 1292.75 | 1292.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 1448.10 | 1456.01 | 1407.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-11 10:15:00 | 1472.35 | 1448.71 | 1411.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1541.10 | 1582.52 | 1535.57 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-04 10:15:00 | 1516.90 | 1581.87 | 1535.48 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 1490.30 | 1549.94 | 1550.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 11:15:00 | 1476.80 | 1546.11 | 1548.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1527.40 | 1499.88 | 1519.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 1484.40 | 1506.83 | 1519.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-08 10:15:00 | 1503.50 | 1483.75 | 1503.14 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 1622.80 | 1515.32 | 1515.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 1633.10 | 1517.56 | 1516.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1681.40 | 1682.13 | 1636.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-18 13:15:00 | 1702.00 | 1682.73 | 1638.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-04 09:15:00 | 1657.10 | 1692.27 | 1658.21 | Close below EMA400 |

### Cycle 12 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 1563.20 | 1682.91 | 1682.94 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 11:15:00 | 1726.60 | 1682.78 | 1682.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 1742.50 | 1686.43 | 1684.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 09:15:00 | 1700.00 | 1700.31 | 1692.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-16 10:15:00 | 1702.50 | 1700.33 | 1692.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-16 11:15:00 | 1686.50 | 1700.19 | 1692.62 | Close below EMA400 |

### Cycle 14 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 1541.90 | 1691.45 | 1691.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1511.30 | 1673.56 | 1682.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1526.80 | 1507.14 | 1575.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 14:15:00 | 1508.00 | 1515.77 | 1572.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-16 09:15:00 | 1597.30 | 1518.77 | 1571.25 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-03 09:15:00 | 1215.50 | 2023-11-03 11:15:00 | 1146.70 | EXIT_EMA400 | -68.80 |
| SELL | 2023-12-05 09:15:00 | 1137.25 | 2023-12-06 09:15:00 | 1148.15 | EXIT_EMA400 | -10.90 |
| BUY | 2024-01-24 12:15:00 | 1225.75 | 2024-01-24 15:15:00 | 1263.15 | TARGET | 37.40 |
| BUY | 2024-01-11 09:15:00 | 1230.35 | 2024-01-30 14:15:00 | 1212.30 | EXIT_EMA400 | -18.05 |
| BUY | 2024-05-31 11:15:00 | 1245.25 | 2024-06-10 09:15:00 | 1353.72 | TARGET | 108.47 |
| BUY | 2024-06-05 10:15:00 | 1273.75 | 2024-06-14 09:15:00 | 1453.68 | TARGET | 179.93 |
| SELL | 2024-12-05 10:15:00 | 1284.45 | 2024-12-11 09:15:00 | 1342.90 | EXIT_EMA400 | -58.45 |
| BUY | 2025-04-11 10:15:00 | 1472.35 | 2025-04-21 12:15:00 | 1655.23 | TARGET | 182.88 |
| SELL | 2025-08-26 09:15:00 | 1484.40 | 2025-09-08 10:15:00 | 1503.50 | EXIT_EMA400 | -19.10 |
| BUY | 2025-11-18 13:15:00 | 1702.00 | 2025-12-04 09:15:00 | 1657.10 | EXIT_EMA400 | -44.90 |
| BUY | 2026-02-16 10:15:00 | 1702.50 | 2026-02-16 11:15:00 | 1686.50 | EXIT_EMA400 | -16.00 |
| SELL | 2026-04-13 14:15:00 | 1508.00 | 2026-04-16 09:15:00 | 1597.30 | EXIT_EMA400 | -89.30 |
