# Grasim Industries Ltd. (GRASIM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2794.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 59.70
- **Avg P&L per closed trade:** 19.90

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 11:15:00 | 2649.50 | 2687.92 | 2688.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 09:15:00 | 2583.90 | 2685.28 | 2686.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 2622.10 | 2611.63 | 2642.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-27 11:15:00 | 2598.00 | 2612.14 | 2640.72 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-02 09:15:00 | 2657.35 | 2611.06 | 2637.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 15:15:00 | 2605.50 | 2471.26 | 2471.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 2626.95 | 2472.80 | 2471.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 11:15:00 | 2485.00 | 2509.12 | 2491.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 2581.50 | 2509.41 | 2492.13 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-27 09:15:00 | 2609.00 | 2691.55 | 2639.68 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 2738.70 | 2803.65 | 2803.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 2723.00 | 2800.52 | 2802.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2763.90 | 2762.38 | 2778.39 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 2834.30 | 2787.98 | 2787.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 12:15:00 | 2846.00 | 2793.03 | 2790.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 2813.80 | 2816.82 | 2804.48 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2758.60 | 2796.54 | 2796.63 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 2842.30 | 2796.82 | 2796.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 2857.90 | 2798.38 | 2797.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 2795.60 | 2805.26 | 2801.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-03 09:15:00 | 2829.40 | 2800.04 | 2798.88 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-20 14:15:00 | 2833.90 | 2857.65 | 2834.35 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 2661.50 | 2821.11 | 2821.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 2630.00 | 2793.23 | 2806.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2738.70 | 2670.97 | 2725.46 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-27 11:15:00 | 2598.00 | 2024-12-02 09:15:00 | 2657.35 | EXIT_EMA400 | -59.35 |
| BUY | 2025-04-08 09:15:00 | 2581.50 | 2025-05-27 09:15:00 | 2609.00 | EXIT_EMA400 | 27.50 |
| BUY | 2026-02-03 09:15:00 | 2829.40 | 2026-02-09 13:15:00 | 2920.95 | TARGET | 91.55 |
