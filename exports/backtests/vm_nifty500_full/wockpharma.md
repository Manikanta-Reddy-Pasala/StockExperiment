# Wockhardt Ltd. (WOCKPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1395.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -214.52
- **Avg P&L per closed trade:** -30.65

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 14:15:00 | 224.00 | 234.93 | 234.96 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 11:15:00 | 241.55 | 234.95 | 234.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 09:15:00 | 243.00 | 235.24 | 235.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 14:15:00 | 424.95 | 426.86 | 379.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-18 10:15:00 | 436.85 | 426.85 | 380.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 545.80 | 565.44 | 541.10 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-07 09:15:00 | 540.35 | 564.14 | 541.27 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 13:15:00 | 1193.45 | 1359.45 | 1360.16 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 14:15:00 | 1522.50 | 1353.12 | 1352.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 15:15:00 | 1526.00 | 1354.84 | 1353.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1349.85 | 1390.21 | 1374.31 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 12:15:00 | 1188.85 | 1359.94 | 1360.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 13:15:00 | 1179.40 | 1358.14 | 1359.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 1394.00 | 1352.15 | 1356.16 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 14:15:00 | 1414.90 | 1359.84 | 1359.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 1467.70 | 1369.87 | 1365.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 1370.00 | 1379.76 | 1371.03 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 1258.50 | 1363.12 | 1363.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 1243.90 | 1361.93 | 1362.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 12:15:00 | 1337.50 | 1321.09 | 1338.98 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1463.80 | 1348.10 | 1347.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 1482.20 | 1349.43 | 1348.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 13:15:00 | 1704.50 | 1705.37 | 1620.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-28 09:15:00 | 1738.00 | 1703.97 | 1623.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1633.00 | 1699.74 | 1631.82 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-01 14:15:00 | 1627.70 | 1696.67 | 1631.95 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 1484.60 | 1589.68 | 1589.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 1483.60 | 1583.00 | 1586.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1518.80 | 1513.83 | 1543.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-24 11:15:00 | 1459.70 | 1514.78 | 1535.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-29 13:15:00 | 1538.20 | 1500.84 | 1525.57 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 1449.00 | 1392.80 | 1392.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 10:15:00 | 1461.40 | 1394.09 | 1393.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 13:15:00 | 1408.00 | 1410.17 | 1402.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-16 09:15:00 | 1431.20 | 1406.66 | 1401.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1431.20 | 1406.66 | 1401.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-16 11:15:00 | 1449.70 | 1407.35 | 1401.47 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1412.80 | 1408.43 | 1402.16 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-19 10:15:00 | 1432.00 | 1408.67 | 1402.31 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1404.40 | 1409.28 | 1402.81 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-20 11:15:00 | 1382.00 | 1408.97 | 1402.72 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1354.20 | 1397.12 | 1397.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1348.50 | 1396.27 | 1396.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 1417.40 | 1380.92 | 1388.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 10:15:00 | 1369.60 | 1381.81 | 1388.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-05 11:15:00 | 1397.10 | 1381.96 | 1388.52 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 15:15:00 | 1442.00 | 1330.21 | 1330.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 1445.10 | 1338.48 | 1334.25 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-18 10:15:00 | 436.85 | 2024-02-26 14:15:00 | 605.53 | TARGET | 168.68 |
| BUY | 2025-07-28 09:15:00 | 1738.00 | 2025-08-01 14:15:00 | 1627.70 | EXIT_EMA400 | -110.30 |
| SELL | 2025-09-24 11:15:00 | 1459.70 | 2025-09-29 13:15:00 | 1538.20 | EXIT_EMA400 | -78.50 |
| BUY | 2026-01-16 09:15:00 | 1431.20 | 2026-01-20 11:15:00 | 1382.00 | EXIT_EMA400 | -49.20 |
| BUY | 2026-01-16 11:15:00 | 1449.70 | 2026-01-20 11:15:00 | 1382.00 | EXIT_EMA400 | -67.70 |
| BUY | 2026-01-19 10:15:00 | 1432.00 | 2026-01-20 11:15:00 | 1382.00 | EXIT_EMA400 | -50.00 |
| SELL | 2026-02-05 10:15:00 | 1369.60 | 2026-02-05 11:15:00 | 1397.10 | EXIT_EMA400 | -27.50 |
