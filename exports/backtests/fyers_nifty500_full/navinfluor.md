# Navin Fluorine International Ltd. (NAVINFLUOR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 6815.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -338.04
- **Avg P&L per closed trade:** -42.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 14:15:00 | 3300.25 | 3486.47 | 3486.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 09:15:00 | 3294.95 | 3472.82 | 3479.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 3398.00 | 3382.90 | 3422.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-11 13:15:00 | 3316.00 | 3380.38 | 3419.14 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-24 09:15:00 | 3415.10 | 3335.49 | 3383.14 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 12:15:00 | 3501.70 | 3386.90 | 3386.44 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 3280.70 | 3385.27 | 3385.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 3266.90 | 3382.09 | 3384.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 12:15:00 | 3379.20 | 3368.79 | 3376.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-21 09:15:00 | 3299.35 | 3367.33 | 3376.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 3408.15 | 3359.20 | 3371.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 13:15:00 | 3468.00 | 3381.94 | 3381.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 3501.35 | 3384.89 | 3383.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 12:15:00 | 3485.00 | 3487.88 | 3445.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-16 13:15:00 | 3512.00 | 3488.12 | 3446.01 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-19 09:15:00 | 3407.25 | 3486.51 | 3448.63 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 11:15:00 | 3200.00 | 3419.93 | 3420.38 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 12:15:00 | 3862.00 | 3417.54 | 3416.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 14:15:00 | 3914.35 | 3605.84 | 3535.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 3947.90 | 3950.16 | 3789.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-05 15:15:00 | 4061.00 | 3910.16 | 3805.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 3975.25 | 4096.25 | 3974.77 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-03 13:15:00 | 4094.65 | 4095.20 | 3975.45 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 3774.65 | 4093.15 | 3980.30 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 4644.00 | 4807.09 | 4807.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 4617.60 | 4803.51 | 4805.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 10:15:00 | 4726.00 | 4710.56 | 4751.09 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 5110.00 | 4783.00 | 4782.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 5151.60 | 4797.71 | 4789.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 5642.50 | 5695.26 | 5451.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-10 09:15:00 | 5807.00 | 5686.51 | 5463.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 5792.50 | 5845.07 | 5693.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-09 10:15:00 | 5835.00 | 5844.97 | 5694.64 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 5706.50 | 5842.08 | 5696.16 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-12 09:15:00 | 5789.50 | 5840.29 | 5696.71 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 5761.00 | 5896.52 | 5754.33 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-21 10:15:00 | 5688.50 | 5894.45 | 5754.00 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-11 13:15:00 | 3316.00 | 2024-09-24 09:15:00 | 3415.10 | EXIT_EMA400 | -99.10 |
| SELL | 2024-11-21 09:15:00 | 3299.35 | 2024-11-25 09:15:00 | 3408.15 | EXIT_EMA400 | -108.80 |
| BUY | 2024-12-16 13:15:00 | 3512.00 | 2024-12-19 09:15:00 | 3407.25 | EXIT_EMA400 | -104.75 |
| BUY | 2025-03-05 15:15:00 | 4061.00 | 2025-04-07 09:15:00 | 3774.65 | EXIT_EMA400 | -286.35 |
| BUY | 2025-04-03 13:15:00 | 4094.65 | 2025-04-07 09:15:00 | 3774.65 | EXIT_EMA400 | -320.00 |
| BUY | 2026-01-12 09:15:00 | 5789.50 | 2026-01-14 14:15:00 | 6067.87 | TARGET | 278.37 |
| BUY | 2026-01-09 10:15:00 | 5835.00 | 2026-01-19 10:15:00 | 6256.09 | TARGET | 421.09 |
| BUY | 2025-12-10 09:15:00 | 5807.00 | 2026-01-21 10:15:00 | 5688.50 | EXIT_EMA400 | -118.50 |
