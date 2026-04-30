# Reliance Industries Ltd. (RELIANCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1436.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -130.60
- **Avg P&L per closed trade:** -21.77

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 1470.23 | 1489.25 | 1489.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 1466.23 | 1489.02 | 1489.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1487.48 | 1484.49 | 1486.72 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 09:15:00 | 1497.68 | 1488.44 | 1488.43 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 1480.78 | 1488.38 | 1488.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 1475.08 | 1488.25 | 1488.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 13:15:00 | 1308.55 | 1306.88 | 1351.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 13:15:00 | 1299.00 | 1309.08 | 1345.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-17 09:15:00 | 1288.80 | 1252.89 | 1282.39 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 1300.00 | 1249.14 | 1248.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 14:15:00 | 1302.30 | 1249.67 | 1249.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1475.70 | 1481.30 | 1439.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-16 11:15:00 | 1482.80 | 1481.30 | 1439.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1446.90 | 1480.53 | 1442.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-21 11:15:00 | 1437.90 | 1479.76 | 1442.89 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1384.70 | 1423.44 | 1423.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1380.50 | 1421.69 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-20 15:15:00 | 1412.00 | 1413.77 | 1418.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1412.00 | 1413.77 | 1418.00 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-21 09:15:00 | 1429.90 | 1413.93 | 1418.06 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 1459.00 | 1399.01 | 1398.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1483.00 | 1401.36 | 1400.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.17 | 1511.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-31 09:15:00 | 1548.00 | 1541.24 | 1511.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.51 | 1520.74 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-06 10:15:00 | 1510.50 | 1550.11 | 1520.69 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.20 | 1501.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 1370.70 | 1457.67 | 1477.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1449.75 | 1471.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-06 09:15:00 | 1437.50 | 1449.54 | 1469.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1463.40 | 1450.33 | 1468.71 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-10 11:15:00 | 1459.80 | 1450.57 | 1468.65 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-11 14:15:00 | 1468.90 | 1451.63 | 1468.31 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-09 13:15:00 | 1299.00 | 2025-01-17 09:15:00 | 1288.80 | EXIT_EMA400 | 10.20 |
| BUY | 2025-07-16 11:15:00 | 1482.80 | 2025-07-21 11:15:00 | 1437.90 | EXIT_EMA400 | -44.90 |
| SELL | 2025-08-20 15:15:00 | 1412.00 | 2025-08-21 09:15:00 | 1429.90 | EXIT_EMA400 | -17.90 |
| BUY | 2025-12-31 09:15:00 | 1548.00 | 2026-01-06 10:15:00 | 1510.50 | EXIT_EMA400 | -37.50 |
| SELL | 2026-02-06 09:15:00 | 1437.50 | 2026-02-11 14:15:00 | 1468.90 | EXIT_EMA400 | -31.40 |
| SELL | 2026-02-10 11:15:00 | 1459.80 | 2026-02-11 14:15:00 | 1468.90 | EXIT_EMA400 | -9.10 |
