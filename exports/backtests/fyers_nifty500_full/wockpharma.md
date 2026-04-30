# Wockhardt Ltd. (WOCKPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1395.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -162.57
- **Avg P&L per closed trade:** -32.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 12:15:00 | 1203.05 | 1363.94 | 1364.47 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 15:15:00 | 1526.00 | 1355.93 | 1355.79 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 11:15:00 | 1190.95 | 1362.11 | 1362.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 13:15:00 | 1179.40 | 1358.59 | 1360.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 1394.80 | 1352.57 | 1357.51 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 1402.60 | 1361.96 | 1361.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 1418.20 | 1363.59 | 1362.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 1370.00 | 1379.96 | 1371.98 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 1264.90 | 1364.31 | 1364.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 1258.50 | 1363.26 | 1364.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 12:15:00 | 1337.50 | 1321.20 | 1339.62 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 1484.30 | 1349.49 | 1348.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1512.00 | 1357.83 | 1353.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 13:15:00 | 1704.50 | 1705.39 | 1620.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-28 09:15:00 | 1738.00 | 1704.02 | 1623.68 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1633.00 | 1699.85 | 1632.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-01 14:15:00 | 1627.50 | 1696.78 | 1632.12 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 1484.60 | 1589.76 | 1590.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 1483.60 | 1583.07 | 1586.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1517.50 | 1513.87 | 1543.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-24 11:15:00 | 1460.00 | 1514.79 | 1535.14 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-29 13:15:00 | 1538.20 | 1500.86 | 1525.62 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 1449.00 | 1392.73 | 1392.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 10:15:00 | 1461.50 | 1394.01 | 1393.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 12:15:00 | 1410.10 | 1410.12 | 1401.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-16 10:15:00 | 1434.20 | 1406.91 | 1401.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1412.80 | 1408.35 | 1402.08 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-19 10:15:00 | 1432.00 | 1408.59 | 1402.23 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1404.40 | 1409.21 | 1402.73 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-20 11:15:00 | 1381.20 | 1408.89 | 1402.64 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1354.20 | 1396.97 | 1397.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1348.50 | 1396.13 | 1396.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 11:15:00 | 1379.90 | 1379.16 | 1387.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-02 09:15:00 | 1344.10 | 1383.47 | 1387.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-07 09:15:00 | 1307.00 | 1261.25 | 1304.19 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 1458.90 | 1329.93 | 1329.66 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-07-28 09:15:00 | 1738.00 | 2025-08-01 14:15:00 | 1627.50 | EXIT_EMA400 | -110.50 |
| SELL | 2025-09-24 11:15:00 | 1460.00 | 2025-09-29 13:15:00 | 1538.20 | EXIT_EMA400 | -78.20 |
| BUY | 2026-01-16 10:15:00 | 1434.20 | 2026-01-20 11:15:00 | 1381.20 | EXIT_EMA400 | -53.00 |
| BUY | 2026-01-19 10:15:00 | 1432.00 | 2026-01-20 11:15:00 | 1381.20 | EXIT_EMA400 | -50.80 |
| SELL | 2026-03-02 09:15:00 | 1344.10 | 2026-03-13 12:15:00 | 1214.17 | TARGET | 129.93 |
