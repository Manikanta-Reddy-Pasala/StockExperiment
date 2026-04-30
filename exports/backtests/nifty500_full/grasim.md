# Grasim Industries Ltd. (GRASIM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2794.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -65.90
- **Avg P&L per closed trade:** -16.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 2584.25 | 2685.44 | 2685.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 10:15:00 | 2575.00 | 2684.34 | 2684.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 2622.10 | 2611.71 | 2641.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-27 11:15:00 | 2598.10 | 2612.25 | 2639.94 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-02 09:15:00 | 2658.10 | 2611.11 | 2636.83 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 2626.95 | 2473.04 | 2472.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 2641.20 | 2493.19 | 2482.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2509.00 | 2509.44 | 2491.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 2581.50 | 2509.55 | 2492.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-27 09:15:00 | 2609.00 | 2691.63 | 2639.85 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 2738.70 | 2803.62 | 2803.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 2719.50 | 2799.80 | 2801.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2763.90 | 2762.47 | 2778.41 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 2834.30 | 2788.15 | 2787.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 12:15:00 | 2846.00 | 2793.15 | 2790.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 2813.80 | 2816.80 | 2804.48 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2758.60 | 2796.58 | 2796.66 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 2842.30 | 2796.78 | 2796.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 2858.10 | 2798.35 | 2797.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 2735.90 | 2804.38 | 2800.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-02 14:15:00 | 2772.80 | 2801.63 | 2799.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 2772.80 | 2801.63 | 2799.58 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-02 15:15:00 | 2773.60 | 2801.35 | 2799.45 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 2661.70 | 2821.62 | 2821.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 2628.50 | 2793.62 | 2806.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2738.70 | 2671.23 | 2725.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 2696.20 | 2686.31 | 2728.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-13 10:15:00 | 2730.40 | 2686.75 | 2728.29 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-27 11:15:00 | 2598.10 | 2024-12-02 09:15:00 | 2658.10 | EXIT_EMA400 | -60.00 |
| BUY | 2025-04-08 09:15:00 | 2581.50 | 2025-05-27 09:15:00 | 2609.00 | EXIT_EMA400 | 27.50 |
| BUY | 2026-02-02 14:15:00 | 2772.80 | 2026-02-02 15:15:00 | 2773.60 | EXIT_EMA400 | 0.80 |
| SELL | 2026-04-13 09:15:00 | 2696.20 | 2026-04-13 10:15:00 | 2730.40 | EXIT_EMA400 | -34.20 |
