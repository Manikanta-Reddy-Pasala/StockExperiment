# Bharti Airtel Ltd. (BHARTIARTL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1886.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -43.16
- **Avg P&L per closed trade:** -10.79

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 1572.00 | 1598.17 | 1598.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 12:15:00 | 1568.80 | 1597.87 | 1598.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1614.00 | 1593.56 | 1595.80 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 15:15:00 | 1642.00 | 1598.35 | 1598.14 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 11:15:00 | 1578.20 | 1598.05 | 1598.14 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 09:15:00 | 1635.10 | 1598.27 | 1598.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 11:15:00 | 1671.00 | 1599.34 | 1598.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 1602.90 | 1607.43 | 1603.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-27 09:15:00 | 1618.30 | 1603.35 | 1601.71 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1618.30 | 1603.35 | 1601.71 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-27 13:15:00 | 1598.55 | 1603.56 | 1601.86 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 11:15:00 | 1590.25 | 1600.48 | 1600.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 1585.95 | 1600.03 | 1600.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 1603.60 | 1599.70 | 1600.10 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 12:15:00 | 1617.25 | 1600.50 | 1600.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 11:15:00 | 1622.20 | 1601.43 | 1600.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-14 13:15:00 | 1586.80 | 1601.30 | 1600.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-15 11:15:00 | 1610.60 | 1601.38 | 1600.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 1610.60 | 1601.38 | 1600.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-16 09:15:00 | 1621.05 | 1601.80 | 1601.17 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 1611.20 | 1615.26 | 1608.83 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-27 12:15:00 | 1602.00 | 1615.10 | 1608.82 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2002.50 | 2056.66 | 2056.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 1997.40 | 2055.06 | 2055.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 2024.20 | 2024.05 | 2038.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 09:15:00 | 2009.30 | 2023.97 | 2037.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 2033.40 | 2022.62 | 2036.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-06 14:15:00 | 2042.60 | 2022.82 | 2036.43 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-27 09:15:00 | 1618.30 | 2024-12-27 13:15:00 | 1598.55 | EXIT_EMA400 | -19.75 |
| BUY | 2025-01-15 11:15:00 | 1610.60 | 2025-01-20 09:15:00 | 1639.54 | TARGET | 28.94 |
| BUY | 2025-01-16 09:15:00 | 1621.05 | 2025-01-27 12:15:00 | 1602.00 | EXIT_EMA400 | -19.05 |
| SELL | 2026-02-05 09:15:00 | 2009.30 | 2026-02-06 14:15:00 | 2042.60 | EXIT_EMA400 | -33.30 |
