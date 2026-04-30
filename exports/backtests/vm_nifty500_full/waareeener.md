# Waaree Energies Ltd. (WAAREEENER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-10-28 09:15:00 → 2026-04-30 15:30:00 (2582 bars)
- **Last close:** 3118.80
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
- **Total realized P&L (per unit):** 100.29
- **Avg P&L per closed trade:** 33.43

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 2408.00 | 2733.61 | 2734.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 11:15:00 | 2369.05 | 2714.29 | 2725.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 2231.20 | 2228.49 | 2345.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-03 11:15:00 | 2213.75 | 2294.00 | 2349.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-17 09:15:00 | 2314.00 | 2239.22 | 2304.31 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 15:15:00 | 2839.90 | 2359.51 | 2357.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 10:15:00 | 2856.30 | 2534.34 | 2464.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 2800.00 | 2802.50 | 2686.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 14:15:00 | 2831.00 | 2802.66 | 2689.09 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-18 10:15:00 | 2697.20 | 2804.21 | 2699.21 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 3176.00 | 3324.33 | 3324.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 3164.40 | 3322.74 | 3323.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 3076.00 | 3074.15 | 3168.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-22 15:15:00 | 3055.00 | 3073.85 | 3166.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 3102.00 | 2750.52 | 2886.74 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 09:15:00 | 3132.00 | 2904.04 | 2903.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 10:15:00 | 3195.10 | 2978.27 | 2945.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 3144.50 | 3239.71 | 3116.93 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-03 11:15:00 | 2213.75 | 2025-04-17 09:15:00 | 2314.00 | EXIT_EMA400 | -100.25 |
| BUY | 2025-06-13 14:15:00 | 2831.00 | 2025-06-18 10:15:00 | 2697.20 | EXIT_EMA400 | -133.80 |
| SELL | 2025-12-22 15:15:00 | 3055.00 | 2026-01-05 13:15:00 | 2720.66 | TARGET | 334.34 |
