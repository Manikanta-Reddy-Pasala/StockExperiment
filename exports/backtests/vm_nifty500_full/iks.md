# Inventurus Knowledge Solutions Ltd. (IKS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-12-19 09:15:00 → 2026-04-30 15:30:00 (2337 bars)
- **Last close:** 1654.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| EXIT | 2 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -120.80
- **Avg P&L per closed trade:** -30.20

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 1758.80 | 1580.83 | 1580.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1795.20 | 1608.42 | 1594.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 14:15:00 | 1625.20 | 1628.24 | 1606.74 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 1567.00 | 1602.26 | 1602.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 1552.00 | 1600.23 | 1601.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 13:15:00 | 1606.80 | 1599.85 | 1601.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-04 09:15:00 | 1576.00 | 1599.58 | 1600.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1576.00 | 1599.58 | 1600.99 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-04 10:15:00 | 1572.40 | 1599.31 | 1600.85 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1588.90 | 1597.97 | 1600.12 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-05 10:15:00 | 1564.90 | 1597.64 | 1599.94 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-06 14:15:00 | 1605.50 | 1594.99 | 1598.46 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 1651.70 | 1552.53 | 1552.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 15:15:00 | 1672.40 | 1555.71 | 1553.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 1590.00 | 1593.70 | 1576.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-24 10:15:00 | 1618.70 | 1593.13 | 1578.28 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-08 12:15:00 | 1601.10 | 1632.82 | 1606.40 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1516.30 | 1642.71 | 1643.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 10:15:00 | 1480.50 | 1641.10 | 1642.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 1392.20 | 1391.37 | 1465.54 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-08-04 09:15:00 | 1576.00 | 2025-08-06 14:15:00 | 1605.50 | EXIT_EMA400 | -29.50 |
| SELL | 2025-08-04 10:15:00 | 1572.40 | 2025-08-06 14:15:00 | 1605.50 | EXIT_EMA400 | -33.10 |
| SELL | 2025-08-05 10:15:00 | 1564.90 | 2025-08-06 14:15:00 | 1605.50 | EXIT_EMA400 | -40.60 |
| BUY | 2025-11-24 10:15:00 | 1618.70 | 2025-12-08 12:15:00 | 1601.10 | EXIT_EMA400 | -17.60 |
