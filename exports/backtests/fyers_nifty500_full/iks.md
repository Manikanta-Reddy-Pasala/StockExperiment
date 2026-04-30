# Inventurus Knowledge Solutions Ltd. (IKS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-12-19 09:15:00 → 2026-04-30 15:15:00 (2354 bars)
- **Last close:** 1650.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -159.10
- **Avg P&L per closed trade:** -31.82

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 1758.90 | 1581.22 | 1580.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1794.40 | 1608.61 | 1594.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 14:15:00 | 1625.00 | 1628.41 | 1606.66 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 1567.00 | 1602.24 | 1602.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 1552.00 | 1600.27 | 1601.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 13:15:00 | 1607.00 | 1599.89 | 1601.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-04 09:15:00 | 1576.00 | 1599.60 | 1600.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1576.00 | 1599.60 | 1600.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-04 10:15:00 | 1572.40 | 1599.33 | 1600.84 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1588.90 | 1597.98 | 1600.11 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-05 10:15:00 | 1565.70 | 1597.66 | 1599.94 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-06 14:15:00 | 1605.50 | 1595.01 | 1598.45 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 1651.70 | 1552.18 | 1551.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 15:15:00 | 1672.40 | 1555.37 | 1553.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 1590.80 | 1593.72 | 1576.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-24 10:15:00 | 1622.60 | 1593.19 | 1578.26 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-08 12:15:00 | 1601.30 | 1632.85 | 1606.39 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1573.50 | 1637.73 | 1637.77 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 1704.30 | 1637.98 | 1637.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1712.00 | 1642.34 | 1640.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1630.30 | 1657.72 | 1648.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-13 15:15:00 | 1665.00 | 1656.92 | 1648.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 1665.00 | 1656.92 | 1648.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-16 09:15:00 | 1629.60 | 1656.65 | 1648.40 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1516.30 | 1641.31 | 1641.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 10:15:00 | 1480.50 | 1639.71 | 1641.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 1392.20 | 1391.33 | 1465.18 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-08-04 09:15:00 | 1576.00 | 2025-08-06 14:15:00 | 1605.50 | EXIT_EMA400 | -29.50 |
| SELL | 2025-08-04 10:15:00 | 1572.40 | 2025-08-06 14:15:00 | 1605.50 | EXIT_EMA400 | -33.10 |
| SELL | 2025-08-05 10:15:00 | 1565.70 | 2025-08-06 14:15:00 | 1605.50 | EXIT_EMA400 | -39.80 |
| BUY | 2025-11-24 10:15:00 | 1622.60 | 2025-12-08 12:15:00 | 1601.30 | EXIT_EMA400 | -21.30 |
| BUY | 2026-02-13 15:15:00 | 1665.00 | 2026-02-16 09:15:00 | 1629.60 | EXIT_EMA400 | -35.40 |
