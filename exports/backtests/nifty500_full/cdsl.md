# Central Depository Services (India) Ltd. (CDSL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1272.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -126.45
- **Avg P&L per closed trade:** -21.08

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 14:15:00 | 828.90 | 903.22 | 903.45 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 15:15:00 | 939.00 | 902.29 | 902.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 941.00 | 904.93 | 903.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-10 11:15:00 | 1003.50 | 1005.75 | 969.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-14 12:15:00 | 1011.30 | 1004.70 | 971.22 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1046.35 | 1036.73 | 1003.81 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 992.03 | 1036.28 | 1003.75 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 1364.00 | 1645.92 | 1646.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 1356.00 | 1640.26 | 1643.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 1205.05 | 1196.95 | 1308.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 14:15:00 | 1182.10 | 1201.24 | 1297.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-21 09:15:00 | 1281.10 | 1201.33 | 1261.54 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 1454.90 | 1289.86 | 1289.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 11:15:00 | 1472.30 | 1299.59 | 1294.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 14:15:00 | 1691.50 | 1700.88 | 1600.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-14 09:15:00 | 1716.90 | 1700.90 | 1601.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1638.00 | 1700.88 | 1629.22 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 13:15:00 | 1627.20 | 1698.80 | 1629.24 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1511.90 | 1590.58 | 1590.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1506.70 | 1589.75 | 1590.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 10:15:00 | 1551.60 | 1551.06 | 1567.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-11 13:15:00 | 1530.90 | 1550.65 | 1565.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-17 10:15:00 | 1568.00 | 1550.37 | 1563.81 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1612.60 | 1559.61 | 1559.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 10:15:00 | 1622.00 | 1570.46 | 1565.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 1581.10 | 1581.91 | 1572.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-31 12:15:00 | 1602.80 | 1582.16 | 1572.35 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-04 09:15:00 | 1546.00 | 1583.15 | 1573.39 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 1491.90 | 1580.66 | 1580.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 1482.50 | 1577.88 | 1579.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 09:15:00 | 1399.70 | 1386.86 | 1440.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 09:15:00 | 1379.20 | 1387.93 | 1437.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 11:15:00 | 1287.10 | 1225.06 | 1286.22 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-14 12:15:00 | 1011.30 | 2024-06-04 10:15:00 | 992.03 | EXIT_EMA400 | -19.27 |
| SELL | 2025-03-26 14:15:00 | 1182.10 | 2025-04-21 09:15:00 | 1281.10 | EXIT_EMA400 | -99.00 |
| BUY | 2025-07-14 09:15:00 | 1716.90 | 2025-07-25 13:15:00 | 1627.20 | EXIT_EMA400 | -89.70 |
| SELL | 2025-09-11 13:15:00 | 1530.90 | 2025-09-17 10:15:00 | 1568.00 | EXIT_EMA400 | -37.10 |
| BUY | 2025-10-31 12:15:00 | 1602.80 | 2025-11-04 09:15:00 | 1546.00 | EXIT_EMA400 | -56.80 |
| SELL | 2026-02-12 09:15:00 | 1379.20 | 2026-03-04 09:15:00 | 1203.78 | TARGET | 175.42 |
