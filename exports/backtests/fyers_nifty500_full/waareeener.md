# Waaree Energies Ltd. (WAAREEENER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-10-28 09:15:00 → 2026-04-30 15:15:00 (2601 bars)
- **Last close:** 3129.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 82.77
- **Avg P&L per closed trade:** 27.59

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 2469.55 | 2737.20 | 2737.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 12:15:00 | 2408.00 | 2733.92 | 2735.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 2231.20 | 2227.75 | 2343.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-03 11:15:00 | 2213.75 | 2293.56 | 2347.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-17 09:15:00 | 2314.00 | 2239.02 | 2303.31 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 14:15:00 | 2837.00 | 2354.46 | 2354.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 15:15:00 | 2896.00 | 2566.29 | 2484.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 2800.00 | 2802.85 | 2686.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 14:15:00 | 2831.00 | 2803.00 | 2689.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-18 10:15:00 | 2696.10 | 2804.50 | 2699.16 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 3164.40 | 3322.93 | 3323.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 11:15:00 | 3154.20 | 3321.26 | 3322.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 3076.00 | 3074.49 | 3168.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-22 15:15:00 | 3060.70 | 3074.24 | 3166.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 3104.00 | 2749.33 | 2881.43 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 3178.00 | 2901.51 | 2901.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 10:15:00 | 3195.00 | 2978.09 | 2944.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 3143.00 | 3248.50 | 3123.30 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-03 11:15:00 | 2213.75 | 2025-04-17 09:15:00 | 2314.00 | EXIT_EMA400 | -100.25 |
| BUY | 2025-06-13 14:15:00 | 2831.00 | 2025-06-18 10:15:00 | 2696.10 | EXIT_EMA400 | -134.90 |
| SELL | 2025-12-22 15:15:00 | 3060.70 | 2026-01-05 09:15:00 | 2742.78 | TARGET | 317.92 |
