# BSE Ltd. (BSE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 3640.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 1 |
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| EXIT | 8 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -287.77
- **Avg P&L per closed trade:** -35.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 13:15:00 | 667.50 | 741.11 | 741.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 14:15:00 | 652.12 | 740.22 | 740.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 13:15:00 | 741.67 | 734.58 | 737.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-22 15:15:00 | 732.35 | 734.93 | 737.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-03-26 09:15:00 | 747.38 | 735.06 | 737.87 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 14:15:00 | 822.33 | 740.94 | 740.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 10:15:00 | 848.33 | 743.50 | 742.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 899.03 | 911.85 | 850.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-29 10:15:00 | 942.48 | 912.16 | 850.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-13 09:15:00 | 861.63 | 919.17 | 871.35 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 11:15:00 | 823.02 | 875.73 | 875.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 12:15:00 | 816.33 | 875.13 | 875.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 09:15:00 | 814.67 | 813.21 | 837.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-06 13:15:00 | 780.75 | 820.60 | 835.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-08 09:15:00 | 861.67 | 818.53 | 833.66 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 12:15:00 | 896.38 | 844.71 | 844.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 902.00 | 846.77 | 845.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 12:15:00 | 1481.62 | 1487.32 | 1362.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-26 13:15:00 | 1503.67 | 1487.48 | 1362.76 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-28 10:15:00 | 1724.25 | 1837.09 | 1727.36 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 11:15:00 | 1437.95 | 1730.82 | 1732.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-06 14:15:00 | 1419.33 | 1721.92 | 1727.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 1591.67 | 1557.46 | 1627.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 09:15:00 | 1541.67 | 1559.28 | 1625.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-28 09:15:00 | 1779.15 | 1558.13 | 1618.34 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 1848.10 | 1666.13 | 1665.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 1872.45 | 1673.65 | 1669.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 2588.70 | 2590.07 | 2368.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 2621.00 | 2590.37 | 2371.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-08 10:15:00 | 2469.90 | 2674.53 | 2498.64 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 2306.30 | 2464.89 | 2465.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 2290.00 | 2459.98 | 2462.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 2366.50 | 2354.14 | 2400.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-10 11:15:00 | 2262.60 | 2352.53 | 2396.34 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-09 10:15:00 | 2277.00 | 2201.47 | 2274.49 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 15:15:00 | 2486.00 | 2325.78 | 2325.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 2500.00 | 2327.51 | 2326.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 2686.50 | 2734.65 | 2611.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-12 15:15:00 | 2796.10 | 2695.21 | 2651.64 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 2676.00 | 2722.23 | 2672.24 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-20 13:15:00 | 2665.40 | 2720.76 | 2672.49 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-22 15:15:00 | 732.35 | 2024-03-26 09:15:00 | 747.38 | EXIT_EMA400 | -15.03 |
| BUY | 2024-04-29 10:15:00 | 942.48 | 2024-05-13 09:15:00 | 861.63 | EXIT_EMA400 | -80.85 |
| SELL | 2024-08-06 13:15:00 | 780.75 | 2024-08-08 09:15:00 | 861.67 | EXIT_EMA400 | -80.92 |
| BUY | 2024-11-26 13:15:00 | 1503.67 | 2024-12-19 13:15:00 | 1926.38 | TARGET | 422.72 |
| SELL | 2025-03-25 09:15:00 | 1541.67 | 2025-03-28 09:15:00 | 1779.15 | EXIT_EMA400 | -237.48 |
| BUY | 2025-06-20 10:15:00 | 2621.00 | 2025-07-08 10:15:00 | 2469.90 | EXIT_EMA400 | -151.10 |
| SELL | 2025-09-10 11:15:00 | 2262.60 | 2025-10-09 10:15:00 | 2277.00 | EXIT_EMA400 | -14.40 |
| BUY | 2026-01-12 15:15:00 | 2796.10 | 2026-01-20 13:15:00 | 2665.40 | EXIT_EMA400 | -130.70 |
