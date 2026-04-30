# Reliance Industries Ltd. (RELIANCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1430.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / EMA400 exits:** 4 / 4
- **Total realized P&L (per unit):** 73.64
- **Avg P&L per closed trade:** 9.21

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 15:15:00 | 1227.88 | 1192.19 | 1192.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 09:15:00 | 1229.95 | 1192.56 | 1192.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 11:15:00 | 1453.57 | 1456.43 | 1405.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-27 09:15:00 | 1475.05 | 1448.73 | 1414.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-03 11:15:00 | 1438.70 | 1464.77 | 1445.26 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 1464.88 | 1493.99 | 1494.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 1453.20 | 1493.59 | 1493.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1487.90 | 1484.61 | 1488.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-30 14:15:00 | 1475.07 | 1488.33 | 1489.98 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-17 09:15:00 | 1288.80 | 1252.83 | 1282.61 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 1300.10 | 1249.12 | 1249.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 14:15:00 | 1302.20 | 1249.65 | 1249.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1475.70 | 1481.25 | 1439.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-16 11:15:00 | 1483.00 | 1481.25 | 1439.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1446.90 | 1480.49 | 1442.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-21 11:15:00 | 1437.70 | 1479.72 | 1442.87 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1384.70 | 1423.43 | 1423.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1381.10 | 1421.69 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-20 15:15:00 | 1411.30 | 1413.76 | 1417.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1411.30 | 1413.76 | 1417.99 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-21 09:15:00 | 1429.80 | 1413.92 | 1418.05 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 1459.00 | 1399.06 | 1398.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1482.40 | 1401.41 | 1400.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.16 | 1511.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-31 09:15:00 | 1548.00 | 1541.23 | 1511.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.52 | 1520.75 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-06 10:15:00 | 1511.20 | 1550.13 | 1520.70 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.19 | 1501.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1363.30 | 1460.02 | 1478.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1454.50 | 1474.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 13:15:00 | 1441.00 | 1454.07 | 1473.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1464.40 | 1455.06 | 1470.87 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-12 11:15:00 | 1457.20 | 1455.15 | 1470.76 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 11:15:00 | 1424.70 | 1411.11 | 1429.49 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-25 13:15:00 | 1419.80 | 1411.32 | 1429.41 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1387.00 | 1363.52 | 1388.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-28 15:15:00 | 1394.00 | 1363.83 | 1388.18 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-03-27 09:15:00 | 1475.05 | 2024-05-03 11:15:00 | 1438.70 | EXIT_EMA400 | -36.35 |
| SELL | 2024-09-30 14:15:00 | 1475.07 | 2024-10-03 11:15:00 | 1430.37 | TARGET | 44.70 |
| BUY | 2025-07-16 11:15:00 | 1483.00 | 2025-07-21 11:15:00 | 1437.70 | EXIT_EMA400 | -45.30 |
| SELL | 2025-08-20 15:15:00 | 1411.30 | 2025-08-21 09:15:00 | 1429.80 | EXIT_EMA400 | -18.50 |
| BUY | 2025-12-31 09:15:00 | 1548.00 | 2026-01-06 10:15:00 | 1511.20 | EXIT_EMA400 | -36.80 |
| SELL | 2026-02-12 11:15:00 | 1457.20 | 2026-02-13 13:15:00 | 1416.51 | TARGET | 40.69 |
| SELL | 2026-02-05 13:15:00 | 1441.00 | 2026-03-02 13:15:00 | 1344.63 | TARGET | 96.37 |
| SELL | 2026-03-25 13:15:00 | 1419.80 | 2026-03-27 09:15:00 | 1390.97 | TARGET | 28.83 |
