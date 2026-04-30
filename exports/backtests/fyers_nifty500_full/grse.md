# Garden Reach Shipbuilders & Engineers Ltd. (GRSE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2946.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -165.78
- **Avg P&L per closed trade:** -23.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 09:15:00 | 1784.30 | 1907.64 | 1908.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 1740.00 | 1889.08 | 1898.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 12:15:00 | 1858.60 | 1856.30 | 1879.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-23 09:15:00 | 1816.00 | 1855.92 | 1879.32 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-15 10:15:00 | 1815.85 | 1751.66 | 1804.77 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 1720.00 | 1487.18 | 1486.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 1742.55 | 1533.82 | 1511.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1463.95 | 1550.37 | 1521.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 15:15:00 | 1538.00 | 1547.32 | 1520.93 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 1538.00 | 1547.32 | 1520.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-08 09:15:00 | 1561.00 | 1547.46 | 1521.13 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-04-09 09:15:00 | 1519.45 | 1547.92 | 1522.27 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 2442.30 | 2624.58 | 2625.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 2432.20 | 2619.19 | 2622.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 2532.10 | 2506.86 | 2554.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-16 15:15:00 | 2435.00 | 2505.26 | 2549.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-17 09:15:00 | 2561.00 | 2505.81 | 2549.61 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 2724.70 | 2579.87 | 2579.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 2749.70 | 2588.04 | 2583.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 2607.00 | 2618.17 | 2600.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 12:15:00 | 2623.60 | 2613.68 | 2599.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 2623.60 | 2613.68 | 2599.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-16 10:15:00 | 2599.00 | 2613.65 | 2600.17 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 2545.90 | 2593.83 | 2594.06 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 2728.10 | 2594.75 | 2594.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 2780.00 | 2601.25 | 2597.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 2715.00 | 2718.74 | 2667.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-25 09:15:00 | 2774.00 | 2719.18 | 2668.14 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-02 10:15:00 | 2654.00 | 2732.53 | 2683.83 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 2405.60 | 2646.31 | 2646.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 2398.20 | 2643.84 | 2645.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 2496.20 | 2492.52 | 2554.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-30 09:15:00 | 2444.10 | 2493.33 | 2550.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-06 10:15:00 | 2538.90 | 2481.89 | 2534.65 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 13:15:00 | 2676.90 | 2424.62 | 2423.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 2745.30 | 2432.54 | 2427.96 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-23 09:15:00 | 1816.00 | 2024-10-04 09:15:00 | 1626.03 | TARGET | 189.97 |
| BUY | 2025-04-07 15:15:00 | 1538.00 | 2025-04-08 09:15:00 | 1589.20 | TARGET | 51.20 |
| BUY | 2025-04-08 09:15:00 | 1561.00 | 2025-04-09 09:15:00 | 1519.45 | EXIT_EMA400 | -41.55 |
| SELL | 2025-09-16 15:15:00 | 2435.00 | 2025-09-17 09:15:00 | 2561.00 | EXIT_EMA400 | -126.00 |
| BUY | 2025-10-15 12:15:00 | 2623.60 | 2025-10-16 10:15:00 | 2599.00 | EXIT_EMA400 | -24.60 |
| BUY | 2025-11-25 09:15:00 | 2774.00 | 2025-12-02 10:15:00 | 2654.00 | EXIT_EMA400 | -120.00 |
| SELL | 2025-12-30 09:15:00 | 2444.10 | 2026-01-06 10:15:00 | 2538.90 | EXIT_EMA400 | -94.80 |
