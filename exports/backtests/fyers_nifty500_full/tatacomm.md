# Tata Communications Ltd. (TATACOMM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1586.90
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
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / EMA400 exits:** 5 / 2
- **Total realized P&L (per unit):** 663.20
- **Avg P&L per closed trade:** 94.74

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 15:15:00 | 1790.65 | 1942.01 | 1942.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1743.60 | 1940.03 | 1941.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 1796.00 | 1794.26 | 1839.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-19 12:15:00 | 1763.00 | 1804.04 | 1829.19 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 09:15:00 | 1563.00 | 1482.03 | 1553.24 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 1644.00 | 1572.55 | 1572.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 10:15:00 | 1664.30 | 1579.42 | 1575.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 1665.60 | 1672.23 | 1638.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 1686.00 | 1666.99 | 1642.26 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-01 10:15:00 | 1691.30 | 1725.63 | 1698.57 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 1591.30 | 1682.11 | 1682.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1570.60 | 1670.26 | 1676.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 1616.50 | 1611.09 | 1637.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-01 10:15:00 | 1595.50 | 1636.28 | 1644.58 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-06 11:15:00 | 1643.90 | 1633.20 | 1642.38 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1868.90 | 1651.23 | 1650.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 1888.10 | 1653.59 | 1651.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 1823.00 | 1824.58 | 1768.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-12 15:15:00 | 1854.00 | 1824.75 | 1769.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1814.90 | 1847.30 | 1811.60 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-08 14:15:00 | 1809.50 | 1846.93 | 1811.59 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 13:15:00 | 1725.90 | 1801.08 | 1801.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 1705.90 | 1782.54 | 1791.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 14:15:00 | 1678.00 | 1647.55 | 1703.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-16 09:15:00 | 1636.10 | 1653.71 | 1700.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1684.10 | 1653.34 | 1692.71 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-23 09:15:00 | 1663.00 | 1653.44 | 1692.57 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1678.00 | 1654.69 | 1692.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-24 09:15:00 | 1643.50 | 1654.58 | 1691.80 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1532.00 | 1470.00 | 1533.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-15 10:15:00 | 1542.10 | 1470.71 | 1533.62 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-19 12:15:00 | 1763.00 | 2025-01-27 14:15:00 | 1564.42 | TARGET | 198.58 |
| BUY | 2025-06-27 09:15:00 | 1686.00 | 2025-07-02 12:15:00 | 1817.23 | TARGET | 131.23 |
| SELL | 2025-10-01 10:15:00 | 1595.50 | 2025-10-06 11:15:00 | 1643.90 | EXIT_EMA400 | -48.40 |
| BUY | 2025-11-12 15:15:00 | 1854.00 | 2025-12-08 14:15:00 | 1809.50 | EXIT_EMA400 | -44.50 |
| SELL | 2026-02-23 09:15:00 | 1663.00 | 2026-03-02 09:15:00 | 1574.30 | TARGET | 88.70 |
| SELL | 2026-02-24 09:15:00 | 1643.50 | 2026-03-04 09:15:00 | 1498.59 | TARGET | 144.91 |
| SELL | 2026-02-16 09:15:00 | 1636.10 | 2026-03-05 13:15:00 | 1443.40 | TARGET | 192.70 |
