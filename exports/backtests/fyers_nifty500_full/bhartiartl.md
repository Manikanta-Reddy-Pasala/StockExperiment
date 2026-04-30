# Bharti Airtel Ltd. (BHARTIARTL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1891.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -58.66
- **Avg P&L per closed trade:** -11.73

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 10:15:00 | 1584.00 | 1599.80 | 1599.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-25 11:15:00 | 1574.80 | 1599.55 | 1599.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1613.65 | 1593.40 | 1596.43 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 12:15:00 | 1626.05 | 1599.31 | 1599.27 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 15:15:00 | 1583.65 | 1599.15 | 1599.20 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 13:15:00 | 1609.75 | 1599.25 | 1599.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 1619.80 | 1599.45 | 1599.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 12:15:00 | 1598.80 | 1599.85 | 1599.55 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 1575.35 | 1599.20 | 1599.25 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 11:15:00 | 1671.40 | 1599.28 | 1599.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 12:15:00 | 1671.50 | 1600.00 | 1599.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 1603.55 | 1607.40 | 1603.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-27 09:15:00 | 1618.30 | 1603.30 | 1602.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1618.30 | 1603.30 | 1602.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-27 13:15:00 | 1598.50 | 1603.52 | 1602.17 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 1588.00 | 1601.00 | 1601.04 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 15:15:00 | 1611.00 | 1600.98 | 1600.97 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 1597.75 | 1600.96 | 1600.96 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 09:15:00 | 1617.00 | 1601.05 | 1601.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 11:15:00 | 1628.60 | 1602.22 | 1601.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 10:15:00 | 1611.50 | 1615.26 | 1609.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-30 14:15:00 | 1641.95 | 1615.31 | 1609.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1614.95 | 1615.58 | 1609.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-02-03 10:15:00 | 1657.95 | 1617.33 | 1611.24 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1617.65 | 1623.88 | 1615.45 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-02-06 15:15:00 | 1621.90 | 1623.86 | 1615.49 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-02-20 09:15:00 | 1633.50 | 1653.54 | 1635.31 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2002.80 | 2056.72 | 2056.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 1997.40 | 2055.12 | 2055.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 10:15:00 | 2022.20 | 2019.76 | 2035.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 09:15:00 | 2008.90 | 2019.88 | 2035.18 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-06 11:15:00 | 2034.10 | 2018.71 | 2033.91 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-27 09:15:00 | 1618.30 | 2024-12-27 13:15:00 | 1598.50 | EXIT_EMA400 | -19.80 |
| BUY | 2025-02-06 15:15:00 | 1621.90 | 2025-02-07 09:15:00 | 1641.14 | TARGET | 19.24 |
| BUY | 2025-01-30 14:15:00 | 1641.95 | 2025-02-20 09:15:00 | 1633.50 | EXIT_EMA400 | -8.45 |
| BUY | 2025-02-03 10:15:00 | 1657.95 | 2025-02-20 09:15:00 | 1633.50 | EXIT_EMA400 | -24.45 |
| SELL | 2026-02-05 09:15:00 | 2008.90 | 2026-02-06 11:15:00 | 2034.10 | EXIT_EMA400 | -25.20 |
