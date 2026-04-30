# Zen Technologies Ltd. (ZENTEC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1673.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -168.60
- **Avg P&L per closed trade:** -42.15

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 14:15:00 | 1741.45 | 2055.02 | 2056.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1677.20 | 2031.92 | 2044.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 1345.40 | 1308.70 | 1502.90 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 1884.50 | 1489.35 | 1489.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1930.30 | 1509.08 | 1499.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 1887.00 | 1900.23 | 1774.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 1926.00 | 1900.30 | 1777.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1850.30 | 1922.41 | 1841.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-11 13:15:00 | 1841.00 | 1919.38 | 1841.28 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 1457.00 | 1807.82 | 1808.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1439.00 | 1747.15 | 1776.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 11:15:00 | 1533.40 | 1507.02 | 1579.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 11:15:00 | 1486.90 | 1533.79 | 1579.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1452.90 | 1403.00 | 1454.11 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-14 11:15:00 | 1459.10 | 1403.55 | 1454.14 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1433.90 | 1355.54 | 1355.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 1438.00 | 1357.91 | 1356.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 15:15:00 | 1365.00 | 1367.54 | 1362.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-16 14:15:00 | 1407.20 | 1366.60 | 1361.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1407.20 | 1366.60 | 1361.70 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-16 15:15:00 | 1419.00 | 1367.13 | 1361.98 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1372.00 | 1381.50 | 1370.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-23 10:15:00 | 1357.40 | 1381.26 | 1370.36 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-20 10:15:00 | 1926.00 | 2025-07-11 13:15:00 | 1841.00 | EXIT_EMA400 | -85.00 |
| SELL | 2025-09-26 11:15:00 | 1486.90 | 2025-11-14 11:15:00 | 1459.10 | EXIT_EMA400 | 27.80 |
| BUY | 2026-03-16 14:15:00 | 1407.20 | 2026-03-23 10:15:00 | 1357.40 | EXIT_EMA400 | -49.80 |
| BUY | 2026-03-16 15:15:00 | 1419.00 | 2026-03-23 10:15:00 | 1357.40 | EXIT_EMA400 | -61.60 |
