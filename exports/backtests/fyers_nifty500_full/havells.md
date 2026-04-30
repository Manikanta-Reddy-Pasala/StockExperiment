# Havells India Ltd. (HAVELLS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1243.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 40.71
- **Avg P&L per closed trade:** 8.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 12:15:00 | 1708.45 | 1904.70 | 1905.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1694.25 | 1897.27 | 1901.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1724.85 | 1714.22 | 1780.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-20 13:15:00 | 1675.60 | 1724.87 | 1754.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1631.20 | 1592.45 | 1645.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-01 13:15:00 | 1656.70 | 1593.09 | 1645.87 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 1599.30 | 1549.09 | 1548.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 1610.70 | 1549.71 | 1549.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 12:15:00 | 1550.50 | 1563.95 | 1557.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 10:15:00 | 1576.20 | 1561.78 | 1556.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-20 14:15:00 | 1559.50 | 1570.21 | 1562.52 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 1500.40 | 1558.72 | 1558.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 1497.80 | 1558.12 | 1558.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 1548.20 | 1547.44 | 1552.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-17 14:15:00 | 1530.80 | 1550.85 | 1553.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-23 09:15:00 | 1551.50 | 1546.64 | 1551.12 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 1585.60 | 1554.72 | 1554.63 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 1525.60 | 1554.40 | 1554.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 10:15:00 | 1515.00 | 1553.26 | 1553.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 10:15:00 | 1547.00 | 1541.25 | 1546.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-22 11:15:00 | 1526.40 | 1541.10 | 1546.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1526.40 | 1541.10 | 1546.67 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-22 12:15:00 | 1554.10 | 1541.23 | 1546.70 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 1572.00 | 1536.20 | 1536.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 1581.50 | 1536.65 | 1536.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1548.40 | 1567.87 | 1555.73 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 1473.00 | 1546.77 | 1546.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1468.50 | 1530.64 | 1537.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 1510.60 | 1508.94 | 1523.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-24 14:15:00 | 1494.40 | 1508.57 | 1523.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 1449.20 | 1427.60 | 1447.60 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-20 13:15:00 | 1675.60 | 2025-02-01 13:15:00 | 1656.70 | EXIT_EMA400 | 18.90 |
| BUY | 2025-05-12 10:15:00 | 1576.20 | 2025-05-20 14:15:00 | 1559.50 | EXIT_EMA400 | -16.70 |
| SELL | 2025-06-17 14:15:00 | 1530.80 | 2025-06-23 09:15:00 | 1551.50 | EXIT_EMA400 | -20.70 |
| SELL | 2025-07-22 11:15:00 | 1526.40 | 2025-07-22 12:15:00 | 1554.10 | EXIT_EMA400 | -27.70 |
| SELL | 2025-10-24 14:15:00 | 1494.40 | 2025-12-08 11:15:00 | 1407.49 | TARGET | 86.91 |
