# Cholamandalam Investment and Finance Company Ltd. (CHOLAFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1570.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -192.95
- **Avg P&L per closed trade:** -32.16

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 1275.90 | 1467.69 | 1468.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 10:15:00 | 1267.60 | 1454.94 | 1461.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 14:15:00 | 1292.00 | 1291.99 | 1345.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 09:15:00 | 1255.75 | 1299.15 | 1336.24 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-03 09:15:00 | 1300.80 | 1248.00 | 1292.67 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 09:15:00 | 1338.80 | 1290.80 | 1290.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 1397.35 | 1299.12 | 1294.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 1448.55 | 1455.98 | 1407.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-11 11:15:00 | 1493.40 | 1449.04 | 1411.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1541.10 | 1582.55 | 1535.52 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-04 10:15:00 | 1516.90 | 1581.90 | 1535.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 1490.50 | 1549.86 | 1550.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 11:15:00 | 1476.80 | 1546.04 | 1548.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1527.40 | 1499.79 | 1519.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 1484.40 | 1506.75 | 1519.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-08 10:15:00 | 1503.50 | 1483.69 | 1503.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 1622.60 | 1515.28 | 1514.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 1633.10 | 1517.52 | 1516.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1681.40 | 1682.10 | 1636.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 10:15:00 | 1715.60 | 1681.70 | 1645.51 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-04 09:15:00 | 1657.10 | 1692.39 | 1658.29 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1611.30 | 1682.53 | 1682.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1569.70 | 1680.14 | 1681.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1687.10 | 1672.27 | 1677.47 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 1719.30 | 1682.30 | 1682.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 1742.80 | 1682.90 | 1682.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 1686.50 | 1697.82 | 1690.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-16 14:15:00 | 1714.70 | 1697.87 | 1691.05 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-19 13:15:00 | 1692.10 | 1700.66 | 1693.18 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 1546.00 | 1689.08 | 1689.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1512.00 | 1672.92 | 1681.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1527.00 | 1506.87 | 1575.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-30 09:15:00 | 1499.30 | 1538.83 | 1567.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 1557.30 | 1538.10 | 1566.67 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-30 15:15:00 | 1570.50 | 1538.68 | 1566.68 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-17 09:15:00 | 1255.75 | 2025-01-03 09:15:00 | 1300.80 | EXIT_EMA400 | -45.05 |
| BUY | 2025-04-11 11:15:00 | 1493.40 | 2025-06-04 10:15:00 | 1516.90 | EXIT_EMA400 | 23.50 |
| SELL | 2025-08-26 09:15:00 | 1484.40 | 2025-09-08 10:15:00 | 1503.50 | EXIT_EMA400 | -19.10 |
| BUY | 2025-11-26 10:15:00 | 1715.60 | 2025-12-04 09:15:00 | 1657.10 | EXIT_EMA400 | -58.50 |
| BUY | 2026-02-16 14:15:00 | 1714.70 | 2026-02-19 13:15:00 | 1692.10 | EXIT_EMA400 | -22.60 |
| SELL | 2026-04-30 09:15:00 | 1499.30 | 2026-04-30 15:15:00 | 1570.50 | EXIT_EMA400 | -71.20 |
