# Garden Reach Shipbuilders & Engineers Ltd. (GRSE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2930.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 3 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 5 / 6
- **Target hits / EMA400 exits:** 5 / 6
- **Total realized P&L (per unit):** -45.73
- **Avg P&L per closed trade:** -4.16

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 11:15:00 | 811.30 | 844.51 | 844.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 802.00 | 843.79 | 844.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 09:15:00 | 847.10 | 842.99 | 843.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-28 14:15:00 | 803.30 | 841.82 | 843.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 831.30 | 836.90 | 840.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-03-05 11:15:00 | 827.00 | 836.73 | 840.26 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 802.75 | 792.63 | 810.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-01 12:15:00 | 817.20 | 792.96 | 810.79 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 14:15:00 | 895.50 | 824.32 | 824.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 915.35 | 845.86 | 836.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 12:15:00 | 910.40 | 913.04 | 878.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-08 09:15:00 | 943.20 | 913.46 | 879.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 2067.80 | 2258.49 | 1984.12 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-07 13:15:00 | 2170.30 | 2253.71 | 1987.14 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1999.30 | 2229.24 | 1996.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-13 09:15:00 | 1973.50 | 2214.31 | 1996.64 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 11:15:00 | 1792.80 | 1905.35 | 1905.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 14:15:00 | 1779.80 | 1901.80 | 1903.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 12:15:00 | 1859.90 | 1856.33 | 1879.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-23 09:15:00 | 1816.25 | 1855.95 | 1878.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-15 10:15:00 | 1817.00 | 1751.57 | 1804.14 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 1720.00 | 1486.92 | 1486.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 1742.05 | 1533.66 | 1511.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1464.25 | 1550.24 | 1521.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 15:15:00 | 1523.50 | 1547.07 | 1520.71 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 1523.50 | 1547.07 | 1520.71 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-08 09:15:00 | 1561.00 | 1547.21 | 1520.91 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-04-09 09:15:00 | 1519.45 | 1547.70 | 1522.07 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 2442.30 | 2624.62 | 2625.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 2432.20 | 2619.22 | 2622.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 2533.00 | 2506.82 | 2554.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-16 14:15:00 | 2446.10 | 2505.91 | 2550.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-17 09:15:00 | 2561.00 | 2505.76 | 2549.58 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 2724.80 | 2579.70 | 2579.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 2749.70 | 2587.86 | 2583.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 2607.50 | 2618.02 | 2600.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 12:15:00 | 2623.60 | 2613.65 | 2599.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 2623.60 | 2613.65 | 2599.80 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-16 10:15:00 | 2599.00 | 2613.64 | 2600.14 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 2541.00 | 2594.11 | 2594.17 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 2728.10 | 2594.55 | 2594.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 2780.00 | 2601.06 | 2597.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 2715.00 | 2718.63 | 2667.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-25 09:15:00 | 2774.00 | 2719.07 | 2668.05 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-02 10:15:00 | 2654.00 | 2732.35 | 2683.70 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 2405.60 | 2646.24 | 2646.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 2398.20 | 2643.78 | 2645.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 2496.20 | 2492.50 | 2554.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-30 09:15:00 | 2444.10 | 2493.32 | 2550.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-06 10:15:00 | 2537.60 | 2481.80 | 2534.57 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 2727.00 | 2424.85 | 2423.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 2757.00 | 2442.04 | 2432.16 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-05 11:15:00 | 827.00 | 2024-03-12 09:15:00 | 787.21 | TARGET | 39.79 |
| SELL | 2024-02-28 14:15:00 | 803.30 | 2024-03-13 15:15:00 | 683.76 | TARGET | 119.54 |
| BUY | 2024-05-08 09:15:00 | 943.20 | 2024-05-21 09:15:00 | 1134.77 | TARGET | 191.57 |
| BUY | 2024-08-07 13:15:00 | 2170.30 | 2024-08-13 09:15:00 | 1973.50 | EXIT_EMA400 | -196.80 |
| SELL | 2024-09-23 09:15:00 | 1816.25 | 2024-10-04 09:15:00 | 1629.89 | TARGET | 186.36 |
| BUY | 2025-04-07 15:15:00 | 1523.50 | 2025-04-08 09:15:00 | 1531.86 | TARGET | 8.36 |
| BUY | 2025-04-08 09:15:00 | 1561.00 | 2025-04-09 09:15:00 | 1519.45 | EXIT_EMA400 | -41.55 |
| SELL | 2025-09-16 14:15:00 | 2446.10 | 2025-09-17 09:15:00 | 2561.00 | EXIT_EMA400 | -114.90 |
| BUY | 2025-10-15 12:15:00 | 2623.60 | 2025-10-16 10:15:00 | 2599.00 | EXIT_EMA400 | -24.60 |
| BUY | 2025-11-25 09:15:00 | 2774.00 | 2025-12-02 10:15:00 | 2654.00 | EXIT_EMA400 | -120.00 |
| SELL | 2025-12-30 09:15:00 | 2444.10 | 2026-01-06 10:15:00 | 2537.60 | EXIT_EMA400 | -93.50 |
