# Cartrade Tech Ltd. (CARTRADE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1627.00
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
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 1
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 13.91
- **Avg P&L per closed trade:** 4.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 900.00 | 857.61 | 857.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 904.60 | 859.49 | 858.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 10:15:00 | 854.45 | 865.32 | 861.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-19 09:15:00 | 883.10 | 861.89 | 860.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 881.15 | 875.59 | 868.00 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-27 10:15:00 | 889.90 | 875.73 | 868.11 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 869.50 | 875.96 | 868.71 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-29 10:15:00 | 855.85 | 875.76 | 868.64 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 1571.50 | 1602.55 | 1602.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 1559.80 | 1601.85 | 1602.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 12:15:00 | 1602.00 | 1597.75 | 1600.13 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 1657.20 | 1602.46 | 1602.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 1685.00 | 1613.69 | 1608.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 1628.80 | 1628.96 | 1617.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 1638.50 | 1629.07 | 1617.86 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-24 12:15:00 | 1617.30 | 1629.07 | 1618.02 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 12:15:00 | 2519.00 | 2765.23 | 2765.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 2443.40 | 2730.98 | 2748.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1866.00 | 1813.09 | 2015.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-15 14:15:00 | 1782.80 | 1819.01 | 1988.31 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-08-19 09:15:00 | 883.10 | 2024-08-23 09:15:00 | 952.26 | TARGET | 69.16 |
| BUY | 2024-08-27 10:15:00 | 889.90 | 2024-08-29 10:15:00 | 855.85 | EXIT_EMA400 | -34.05 |
| BUY | 2025-06-24 09:15:00 | 1638.50 | 2025-06-24 12:15:00 | 1617.30 | EXIT_EMA400 | -21.20 |
