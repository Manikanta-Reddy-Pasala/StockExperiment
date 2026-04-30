# Central Depository Services (India) Ltd. (CDSL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1273.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -190.12
- **Avg P&L per closed trade:** -38.02

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 1364.00 | 1646.00 | 1646.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 1356.00 | 1640.33 | 1643.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 1196.00 | 1195.23 | 1305.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 1116.60 | 1203.33 | 1278.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-21 09:15:00 | 1281.00 | 1200.76 | 1259.82 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 1456.00 | 1288.07 | 1287.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 11:15:00 | 1472.30 | 1299.46 | 1293.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 14:15:00 | 1691.50 | 1700.87 | 1600.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-14 09:15:00 | 1716.80 | 1700.88 | 1601.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1638.00 | 1700.85 | 1629.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 13:15:00 | 1627.20 | 1698.78 | 1629.07 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1512.00 | 1590.57 | 1590.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1506.70 | 1589.73 | 1590.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 10:15:00 | 1551.60 | 1551.09 | 1567.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-11 13:15:00 | 1530.90 | 1550.68 | 1565.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-17 10:15:00 | 1568.00 | 1550.42 | 1563.79 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1613.20 | 1559.64 | 1559.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 10:15:00 | 1622.00 | 1570.48 | 1565.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 1581.10 | 1581.94 | 1572.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-31 12:15:00 | 1602.80 | 1582.19 | 1572.36 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-04 09:15:00 | 1546.00 | 1583.17 | 1573.40 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 1491.80 | 1580.64 | 1580.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 1482.50 | 1577.86 | 1579.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 09:15:00 | 1399.70 | 1379.94 | 1434.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 09:15:00 | 1379.70 | 1381.95 | 1432.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 1284.50 | 1223.91 | 1284.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 11:15:00 | 1287.20 | 1224.54 | 1284.65 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-07 09:15:00 | 1116.60 | 2025-04-21 09:15:00 | 1281.00 | EXIT_EMA400 | -164.40 |
| BUY | 2025-07-14 09:15:00 | 1716.80 | 2025-07-25 13:15:00 | 1627.20 | EXIT_EMA400 | -89.60 |
| SELL | 2025-09-11 13:15:00 | 1530.90 | 2025-09-17 10:15:00 | 1568.00 | EXIT_EMA400 | -37.10 |
| BUY | 2025-10-31 12:15:00 | 1602.80 | 2025-11-04 09:15:00 | 1546.00 | EXIT_EMA400 | -56.80 |
| SELL | 2026-02-12 09:15:00 | 1379.70 | 2026-03-02 09:15:00 | 1221.92 | TARGET | 157.78 |
