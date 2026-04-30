# BSE Ltd. (BSE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3633.30
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -115.24
- **Avg P&L per closed trade:** -23.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 11:15:00 | 896.65 | 843.76 | 843.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 902.00 | 846.37 | 844.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 12:15:00 | 1481.62 | 1488.18 | 1363.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-26 13:15:00 | 1503.67 | 1488.34 | 1364.19 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-28 10:15:00 | 1724.25 | 1837.08 | 1727.63 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 10:15:00 | 1441.03 | 1733.38 | 1734.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-06 13:15:00 | 1432.28 | 1724.59 | 1730.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 1591.67 | 1557.28 | 1628.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 09:15:00 | 1542.27 | 1559.15 | 1626.35 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-28 09:15:00 | 1779.15 | 1558.02 | 1618.99 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 1848.23 | 1667.91 | 1667.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 1862.07 | 1671.62 | 1668.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 2589.00 | 2590.17 | 2369.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 2621.00 | 2590.46 | 2371.35 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-08 10:15:00 | 2469.50 | 2674.61 | 2498.79 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 2306.30 | 2465.00 | 2465.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 2290.00 | 2460.08 | 2462.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 2366.50 | 2354.16 | 2400.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-10 11:15:00 | 2262.30 | 2352.52 | 2396.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-09 10:15:00 | 2277.00 | 2201.37 | 2274.46 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 15:15:00 | 2486.50 | 2325.56 | 2325.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 2500.00 | 2327.29 | 2326.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 2686.50 | 2734.65 | 2611.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-12 15:15:00 | 2796.10 | 2695.19 | 2651.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 2676.00 | 2722.09 | 2672.15 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-20 13:15:00 | 2665.50 | 2720.65 | 2672.42 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-11-26 13:15:00 | 1503.67 | 2024-12-17 10:15:00 | 1922.11 | TARGET | 418.44 |
| SELL | 2025-03-25 09:15:00 | 1542.27 | 2025-03-28 09:15:00 | 1779.15 | EXIT_EMA400 | -236.88 |
| BUY | 2025-06-20 10:15:00 | 2621.00 | 2025-07-08 10:15:00 | 2469.50 | EXIT_EMA400 | -151.50 |
| SELL | 2025-09-10 11:15:00 | 2262.30 | 2025-10-09 10:15:00 | 2277.00 | EXIT_EMA400 | -14.70 |
| BUY | 2026-01-12 15:15:00 | 2796.10 | 2026-01-20 13:15:00 | 2665.50 | EXIT_EMA400 | -130.60 |
