# Tech Mahindra Ltd. (TECHM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1476.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 105.04
- **Avg P&L per closed trade:** 17.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 1651.60 | 1690.58 | 1690.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1633.20 | 1685.34 | 1687.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 12:15:00 | 1682.95 | 1678.60 | 1683.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 12:15:00 | 1664.60 | 1679.70 | 1684.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 1664.60 | 1679.70 | 1684.09 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-12 11:15:00 | 1684.90 | 1679.55 | 1683.88 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 14:15:00 | 1575.60 | 1503.54 | 1503.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1590.00 | 1505.15 | 1504.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 1536.90 | 1540.70 | 1525.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 13:15:00 | 1565.10 | 1541.85 | 1526.61 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-10 09:15:00 | 1599.70 | 1639.85 | 1604.61 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1460.30 | 1585.43 | 1585.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 1447.50 | 1581.51 | 1583.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 1525.00 | 1520.51 | 1544.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 11:15:00 | 1502.60 | 1520.33 | 1544.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1527.90 | 1514.93 | 1537.09 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-26 14:15:00 | 1500.90 | 1515.70 | 1536.20 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 1528.50 | 1505.38 | 1524.67 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 1546.10 | 1472.93 | 1472.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1574.70 | 1476.65 | 1474.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 1578.30 | 1579.91 | 1546.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-09 09:15:00 | 1588.50 | 1579.96 | 1547.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1628.90 | 1665.17 | 1613.68 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-04 10:15:00 | 1606.40 | 1664.59 | 1613.64 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 13:15:00 | 1441.90 | 1589.59 | 1589.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 14:15:00 | 1439.60 | 1588.09 | 1589.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.70 | 1416.16 | 1473.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-25 15:15:00 | 1403.50 | 1416.80 | 1470.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.81 | 1461.66 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-07 09:15:00 | 1463.80 | 1416.51 | 1461.38 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-11 12:15:00 | 1664.60 | 2025-02-12 11:15:00 | 1684.90 | EXIT_EMA400 | -20.30 |
| BUY | 2025-06-04 13:15:00 | 1565.10 | 2025-06-16 11:15:00 | 1680.56 | TARGET | 115.46 |
| SELL | 2025-08-14 11:15:00 | 1502.60 | 2025-09-10 09:15:00 | 1528.50 | EXIT_EMA400 | -25.90 |
| SELL | 2025-08-26 14:15:00 | 1500.90 | 2025-09-10 09:15:00 | 1528.50 | EXIT_EMA400 | -27.60 |
| BUY | 2026-01-09 09:15:00 | 1588.50 | 2026-01-19 09:15:00 | 1712.19 | TARGET | 123.69 |
| SELL | 2026-03-25 15:15:00 | 1403.50 | 2026-04-07 09:15:00 | 1463.80 | EXIT_EMA400 | -60.30 |
