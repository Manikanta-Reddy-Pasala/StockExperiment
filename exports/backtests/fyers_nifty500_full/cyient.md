# Cyient Ltd. (CYIENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 872.00
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -130.35
- **Avg P&L per closed trade:** -43.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 1685.85 | 1791.38 | 1791.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 12:15:00 | 1674.00 | 1781.54 | 1786.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 11:15:00 | 1778.95 | 1762.11 | 1775.24 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 2003.20 | 1787.37 | 1786.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 11:15:00 | 2009.35 | 1842.23 | 1816.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 15:15:00 | 1987.00 | 1996.43 | 1929.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-25 10:15:00 | 2007.90 | 1996.56 | 1930.49 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-27 09:15:00 | 1932.55 | 1995.48 | 1934.11 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 1765.60 | 1902.37 | 1902.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 1763.90 | 1901.00 | 1902.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 1871.70 | 1853.56 | 1873.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 14:15:00 | 1815.95 | 1863.62 | 1875.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1848.35 | 1839.86 | 1859.97 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-26 09:15:00 | 1893.90 | 1840.80 | 1859.76 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 2018.15 | 1872.21 | 1871.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 2025.00 | 1873.73 | 1872.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 1946.40 | 1960.34 | 1923.77 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 1743.95 | 1901.29 | 1901.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 1738.90 | 1873.41 | 1886.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 11:15:00 | 1342.75 | 1331.79 | 1457.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-21 13:15:00 | 1310.95 | 1331.49 | 1455.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-14 09:15:00 | 1288.00 | 1218.10 | 1287.08 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-25 10:15:00 | 2007.90 | 2024-09-27 09:15:00 | 1932.55 | EXIT_EMA400 | -75.35 |
| SELL | 2024-11-12 14:15:00 | 1815.95 | 2024-11-26 09:15:00 | 1893.90 | EXIT_EMA400 | -77.95 |
| SELL | 2025-03-21 13:15:00 | 1310.95 | 2025-05-14 09:15:00 | 1288.00 | EXIT_EMA400 | 22.95 |
