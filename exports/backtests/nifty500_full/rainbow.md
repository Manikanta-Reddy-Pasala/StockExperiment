# Rainbow Childrens Medicare Ltd. (RAINBOW.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1252.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -139.91
- **Avg P&L per closed trade:** -23.32

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 14:15:00 | 1126.70 | 1202.99 | 1203.12 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 1249.00 | 1203.44 | 1203.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 12:15:00 | 1262.65 | 1206.83 | 1205.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 10:15:00 | 1350.40 | 1351.21 | 1302.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-09 09:15:00 | 1412.55 | 1352.00 | 1304.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-22 09:15:00 | 1275.00 | 1352.04 | 1316.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 09:15:00 | 1264.00 | 1297.31 | 1297.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 11:15:00 | 1250.00 | 1296.41 | 1296.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 12:15:00 | 1290.55 | 1289.25 | 1293.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-05 11:15:00 | 1284.40 | 1291.23 | 1293.86 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-07-05 14:15:00 | 1303.95 | 1291.19 | 1293.80 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 1299.65 | 1240.04 | 1239.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 11:15:00 | 1309.85 | 1240.74 | 1240.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 10:15:00 | 1365.00 | 1382.05 | 1338.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-24 10:15:00 | 1385.10 | 1380.67 | 1340.92 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-23 09:15:00 | 1528.65 | 1588.28 | 1531.79 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 09:15:00 | 1409.75 | 1518.60 | 1518.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1369.60 | 1509.97 | 1514.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 1317.85 | 1313.34 | 1368.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-18 13:15:00 | 1303.15 | 1313.15 | 1367.81 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 1357.70 | 1312.39 | 1359.94 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 15:15:00 | 1360.00 | 1312.86 | 1359.94 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 14:15:00 | 1574.60 | 1383.85 | 1383.20 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 1319.90 | 1391.31 | 1391.54 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 1426.60 | 1385.70 | 1385.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 12:15:00 | 1445.80 | 1392.17 | 1389.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 1519.40 | 1519.96 | 1484.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-01 09:15:00 | 1528.50 | 1519.88 | 1485.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1492.10 | 1519.35 | 1487.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-05 11:15:00 | 1484.80 | 1518.74 | 1487.89 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 1450.30 | 1496.52 | 1496.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 1430.50 | 1494.95 | 1495.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1378.30 | 1375.46 | 1412.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 10:15:00 | 1363.70 | 1375.97 | 1409.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1368.00 | 1350.03 | 1369.52 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-11 14:15:00 | 1378.50 | 1350.31 | 1369.57 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 1289.00 | 1215.16 | 1214.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 1300.00 | 1216.00 | 1215.27 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-09 09:15:00 | 1412.55 | 2024-05-22 09:15:00 | 1275.00 | EXIT_EMA400 | -137.55 |
| SELL | 2024-07-05 11:15:00 | 1284.40 | 2024-07-05 14:15:00 | 1303.95 | EXIT_EMA400 | -19.55 |
| BUY | 2024-10-24 10:15:00 | 1385.10 | 2024-10-31 09:15:00 | 1517.64 | TARGET | 132.54 |
| SELL | 2025-03-18 13:15:00 | 1303.15 | 2025-03-24 15:15:00 | 1360.00 | EXIT_EMA400 | -56.85 |
| BUY | 2025-08-01 09:15:00 | 1528.50 | 2025-08-05 11:15:00 | 1484.80 | EXIT_EMA400 | -43.70 |
| SELL | 2025-10-31 10:15:00 | 1363.70 | 2025-12-11 14:15:00 | 1378.50 | EXIT_EMA400 | -14.80 |
