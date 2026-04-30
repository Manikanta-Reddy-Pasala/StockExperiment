# Zen Technologies Ltd. (ZENTEC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1671.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 249.52
- **Avg P&L per closed trade:** 62.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 14:15:00 | 1742.85 | 2055.01 | 2056.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1679.40 | 2048.21 | 2052.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 1347.00 | 1310.48 | 1505.63 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 1884.50 | 1493.36 | 1491.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1930.30 | 1509.15 | 1499.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 1887.00 | 1900.18 | 1774.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 1926.00 | 1900.23 | 1777.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1850.30 | 1922.33 | 1841.32 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-11 13:15:00 | 1841.00 | 1919.31 | 1841.40 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 1455.00 | 1807.79 | 1808.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1439.00 | 1747.13 | 1776.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 11:15:00 | 1533.40 | 1506.95 | 1579.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 11:15:00 | 1486.90 | 1533.72 | 1579.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1454.00 | 1403.04 | 1454.13 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-14 11:15:00 | 1459.50 | 1403.60 | 1454.16 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1436.90 | 1356.05 | 1355.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 14:15:00 | 1445.50 | 1374.38 | 1366.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 1372.00 | 1381.33 | 1370.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-07 09:15:00 | 1406.90 | 1368.73 | 1365.93 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1406.90 | 1368.73 | 1365.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-04-07 11:15:00 | 1427.90 | 1369.81 | 1366.50 | Buy entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-20 10:15:00 | 1926.00 | 2025-07-11 13:15:00 | 1841.00 | EXIT_EMA400 | -85.00 |
| SELL | 2025-09-26 11:15:00 | 1486.90 | 2025-11-14 11:15:00 | 1459.50 | EXIT_EMA400 | 27.40 |
| BUY | 2026-04-07 09:15:00 | 1406.90 | 2026-04-09 11:15:00 | 1529.82 | TARGET | 122.92 |
| BUY | 2026-04-07 11:15:00 | 1427.90 | 2026-04-17 14:15:00 | 1612.11 | TARGET | 184.21 |
