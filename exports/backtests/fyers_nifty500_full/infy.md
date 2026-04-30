# Infosys Ltd. (INFY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1183.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -234.70
- **Avg P&L per closed trade:** -58.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 1823.00 | 1894.80 | 1894.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 13:15:00 | 1817.65 | 1873.62 | 1881.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1519.10 | 1516.92 | 1605.60 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 11:15:00 | 1605.50 | 1591.48 | 1591.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 09:15:00 | 1641.90 | 1592.54 | 1592.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 1594.20 | 1605.23 | 1599.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-16 12:15:00 | 1608.40 | 1601.09 | 1597.71 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-17 09:15:00 | 1596.00 | 1601.24 | 1597.85 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1560.80 | 1594.94 | 1595.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1552.30 | 1594.52 | 1594.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1496.40 | 1495.73 | 1531.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-29 09:15:00 | 1478.30 | 1500.06 | 1528.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 1524.00 | 1489.08 | 1515.13 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1540.30 | 1500.97 | 1500.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 09:15:00 | 1571.80 | 1506.56 | 1503.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1591.10 | 1606.85 | 1574.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-07 13:15:00 | 1639.50 | 1608.34 | 1578.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1616.70 | 1634.65 | 1606.47 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-01 11:15:00 | 1638.30 | 1634.55 | 1606.70 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.98 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1405.10 | 1587.88 | 1588.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 1399.50 | 1586.00 | 1587.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 1313.10 | 1310.98 | 1382.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-10 09:15:00 | 1289.20 | 1314.93 | 1375.60 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-07-16 12:15:00 | 1608.40 | 2025-07-17 09:15:00 | 1596.00 | EXIT_EMA400 | -12.40 |
| SELL | 2025-08-29 09:15:00 | 1478.30 | 2025-09-10 09:15:00 | 1524.00 | EXIT_EMA400 | -45.70 |
| BUY | 2026-01-07 13:15:00 | 1639.50 | 2026-02-04 09:15:00 | 1550.60 | EXIT_EMA400 | -88.90 |
| BUY | 2026-02-01 11:15:00 | 1638.30 | 2026-02-04 09:15:00 | 1550.60 | EXIT_EMA400 | -87.70 |
