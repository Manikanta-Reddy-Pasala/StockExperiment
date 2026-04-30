# HCL Technologies Ltd. (HCLTECH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1201.40
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
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 200.22
- **Avg P&L per closed trade:** 50.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 14:15:00 | 1710.95 | 1860.66 | 1861.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 11:15:00 | 1709.25 | 1854.92 | 1858.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-25 09:15:00 | 1643.55 | 1614.37 | 1680.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-27 12:15:00 | 1619.95 | 1616.84 | 1675.96 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1569.00 | 1510.51 | 1587.27 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-23 10:15:00 | 1599.00 | 1511.39 | 1587.33 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1650.80 | 1606.38 | 1606.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 1668.90 | 1622.83 | 1615.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1691.70 | 1692.97 | 1665.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 12:15:00 | 1714.50 | 1693.30 | 1666.43 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-10 09:15:00 | 1658.40 | 1692.30 | 1667.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 1534.40 | 1649.25 | 1649.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 1529.50 | 1646.92 | 1648.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 1483.10 | 1476.51 | 1514.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 12:15:00 | 1470.30 | 1478.04 | 1511.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1467.20 | 1443.05 | 1478.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-09 10:15:00 | 1481.00 | 1443.42 | 1478.65 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 1535.20 | 1495.73 | 1495.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1539.50 | 1496.17 | 1495.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1498.10 | 1501.08 | 1498.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-10 09:15:00 | 1532.00 | 1501.90 | 1499.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-05 09:15:00 | 1599.70 | 1637.20 | 1606.62 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 1454.70 | 1625.22 | 1625.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1450.70 | 1587.14 | 1605.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1399.40 | 1395.18 | 1457.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-22 09:15:00 | 1305.20 | 1417.45 | 1449.99 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-27 12:15:00 | 1619.95 | 2025-04-04 09:15:00 | 1451.92 | TARGET | 168.03 |
| BUY | 2025-07-08 12:15:00 | 1714.50 | 2025-07-10 09:15:00 | 1658.40 | EXIT_EMA400 | -56.10 |
| SELL | 2025-09-19 12:15:00 | 1470.30 | 2025-10-09 10:15:00 | 1481.00 | EXIT_EMA400 | -10.70 |
| BUY | 2025-11-10 09:15:00 | 1532.00 | 2025-11-19 10:15:00 | 1630.99 | TARGET | 98.99 |
