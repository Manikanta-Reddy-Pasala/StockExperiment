# Affle 3i Ltd. (AFFLE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1428.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 217.30
- **Avg P&L per closed trade:** 43.46

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 11:15:00 | 1558.25 | 1652.13 | 1652.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 1537.50 | 1647.52 | 1650.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1588.80 | 1578.70 | 1609.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-12 09:15:00 | 1513.10 | 1592.74 | 1612.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-14 10:15:00 | 1611.85 | 1586.83 | 1607.78 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 1616.80 | 1542.48 | 1542.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 13:15:00 | 1622.00 | 1543.27 | 1542.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 10:15:00 | 1558.00 | 1558.16 | 1550.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 11:15:00 | 1593.50 | 1550.79 | 1547.97 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 1843.60 | 1914.78 | 1848.34 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 1918.50 | 1949.77 | 1949.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 1890.00 | 1949.18 | 1949.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 1954.70 | 1945.37 | 1947.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 10:15:00 | 1928.00 | 1945.20 | 1947.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1928.00 | 1945.20 | 1947.44 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-31 15:15:00 | 1925.00 | 1944.66 | 1947.11 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1898.00 | 1944.19 | 1946.87 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-04 09:15:00 | 1826.90 | 1940.73 | 1945.02 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1761.50 | 1706.23 | 1764.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-23 12:15:00 | 1772.10 | 1706.89 | 1764.92 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-12 09:15:00 | 1513.10 | 2025-02-14 10:15:00 | 1611.85 | EXIT_EMA400 | -98.75 |
| BUY | 2025-05-12 11:15:00 | 1593.50 | 2025-05-16 11:15:00 | 1730.09 | TARGET | 136.59 |
| SELL | 2025-10-31 10:15:00 | 1928.00 | 2025-11-04 09:15:00 | 1869.67 | TARGET | 58.33 |
| SELL | 2025-10-31 15:15:00 | 1925.00 | 2025-11-04 09:15:00 | 1858.66 | TARGET | 66.34 |
| SELL | 2025-11-04 09:15:00 | 1826.90 | 2025-12-23 12:15:00 | 1772.10 | EXIT_EMA400 | 54.80 |
