# Aditya Birla Real Estate Ltd. (ABREL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1488.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 3 / 0
- **Target hits / EMA400 exits:** 2 / 1
- **Total realized P&L (per unit):** 536.27
- **Avg P&L per closed trade:** 178.76

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 2475.00 | 2676.96 | 2677.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 2427.15 | 2674.47 | 2676.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 2157.75 | 2150.07 | 2328.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 12:15:00 | 2065.40 | 2149.37 | 2323.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 2068.85 | 1964.22 | 2079.82 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 09:15:00 | 2005.25 | 1970.17 | 2078.87 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-23 14:15:00 | 2018.60 | 1914.18 | 1992.95 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 2190.90 | 2000.02 | 1999.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 2196.20 | 2001.98 | 2000.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 2315.70 | 2333.30 | 2235.79 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 1953.80 | 2196.39 | 2197.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 10:15:00 | 1924.70 | 2174.70 | 2186.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 14:15:00 | 1865.90 | 1852.12 | 1940.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-24 10:15:00 | 1843.10 | 1869.11 | 1932.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1781.60 | 1708.31 | 1787.11 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-29 13:15:00 | 1796.60 | 1709.19 | 1787.15 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 1546.40 | 1344.76 | 1344.50 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-25 09:15:00 | 2005.25 | 2025-04-07 09:15:00 | 1784.39 | TARGET | 220.86 |
| SELL | 2025-02-01 12:15:00 | 2065.40 | 2025-04-23 14:15:00 | 2018.60 | EXIT_EMA400 | 46.80 |
| SELL | 2025-09-24 10:15:00 | 1843.10 | 2025-10-06 13:15:00 | 1574.49 | TARGET | 268.61 |
