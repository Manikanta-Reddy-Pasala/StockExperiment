# Cipla Ltd. (CIPLA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1312.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 9 |
| ENTRY1 | 5 |
| ENTRY2 | 4 |
| EXIT | 5 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 6 / 3
- **Target hits / EMA400 exits:** 6 / 3
- **Total realized P&L (per unit):** 186.74
- **Avg P&L per closed trade:** 20.75

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 1480.20 | 1586.63 | 1586.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1457.85 | 1574.08 | 1580.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 11:15:00 | 1559.80 | 1557.80 | 1571.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-11 12:15:00 | 1549.70 | 1566.14 | 1573.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 1539.40 | 1525.93 | 1546.92 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-29 14:15:00 | 1535.05 | 1526.24 | 1546.76 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1520.60 | 1526.24 | 1546.56 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-02 12:15:00 | 1510.30 | 1526.00 | 1546.14 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 1540.00 | 1525.87 | 1545.47 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-03 13:15:00 | 1534.05 | 1526.11 | 1545.40 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1510.25 | 1491.07 | 1513.46 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-30 10:15:00 | 1523.55 | 1491.39 | 1513.51 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 13:15:00 | 1523.55 | 1471.63 | 1471.52 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 11:15:00 | 1443.90 | 1471.98 | 1472.01 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 1499.95 | 1472.12 | 1472.07 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1410.00 | 1471.58 | 1471.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1381.20 | 1467.95 | 1469.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 1468.00 | 1457.10 | 1464.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-11 10:15:00 | 1454.90 | 1457.08 | 1463.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 1454.90 | 1457.08 | 1463.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 09:15:00 | 1470.90 | 1457.36 | 1463.89 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 1526.90 | 1469.82 | 1469.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 1550.90 | 1475.92 | 1472.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1501.30 | 1501.94 | 1488.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-13 09:15:00 | 1529.60 | 1500.25 | 1489.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1503.60 | 1501.44 | 1490.31 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-15 09:15:00 | 1477.60 | 1500.90 | 1490.42 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1475.30 | 1493.10 | 1493.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1472.20 | 1492.23 | 1492.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 13:15:00 | 1541.60 | 1488.39 | 1490.49 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 13:15:00 | 1567.20 | 1492.84 | 1492.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 14:15:00 | 1573.30 | 1493.64 | 1493.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 1506.70 | 1507.04 | 1500.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-12 13:15:00 | 1520.40 | 1502.73 | 1499.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1541.20 | 1556.94 | 1541.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-23 09:15:00 | 1535.90 | 1556.58 | 1541.14 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 1513.00 | 1531.14 | 1531.16 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1555.20 | 1531.34 | 1531.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 1568.00 | 1531.98 | 1531.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 1546.50 | 1561.42 | 1549.22 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 1524.90 | 1539.68 | 1539.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 13:15:00 | 1518.00 | 1539.12 | 1539.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 1530.00 | 1529.74 | 1533.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-01 09:15:00 | 1524.60 | 1529.71 | 1533.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1524.60 | 1529.71 | 1533.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-02 09:15:00 | 1520.80 | 1529.32 | 1533.35 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1513.60 | 1508.94 | 1517.62 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 1518.40 | 1509.08 | 1517.43 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-11 12:15:00 | 1549.70 | 2024-11-18 09:15:00 | 1478.74 | TARGET | 70.96 |
| SELL | 2024-11-29 14:15:00 | 1535.05 | 2024-12-04 14:15:00 | 1499.91 | TARGET | 35.14 |
| SELL | 2024-12-03 13:15:00 | 1534.05 | 2024-12-04 14:15:00 | 1500.01 | TARGET | 34.04 |
| SELL | 2024-12-02 12:15:00 | 1510.30 | 2024-12-30 10:15:00 | 1523.55 | EXIT_EMA400 | -13.25 |
| SELL | 2025-04-11 10:15:00 | 1454.90 | 2025-04-15 09:15:00 | 1470.90 | EXIT_EMA400 | -16.00 |
| BUY | 2025-05-13 09:15:00 | 1529.60 | 2025-05-15 09:15:00 | 1477.60 | EXIT_EMA400 | -52.00 |
| BUY | 2025-08-12 13:15:00 | 1520.40 | 2025-08-21 14:15:00 | 1583.34 | TARGET | 62.94 |
| SELL | 2025-12-01 09:15:00 | 1524.60 | 2025-12-03 11:15:00 | 1497.34 | TARGET | 27.26 |
| SELL | 2025-12-02 09:15:00 | 1520.80 | 2025-12-30 10:15:00 | 1483.15 | TARGET | 37.65 |
