# InterGlobe Aviation Ltd. (INDIGO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 4295.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 3 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / EMA400 exits:** 4 / 7
- **Total realized P&L (per unit):** 557.01
- **Avg P&L per closed trade:** 50.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 11:15:00 | 2438.95 | 2510.58 | 2510.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-01 09:15:00 | 2425.00 | 2506.83 | 2508.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 10:15:00 | 2502.40 | 2490.86 | 2499.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-09-08 14:15:00 | 2474.30 | 2490.78 | 2499.52 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 2490.05 | 2490.60 | 2499.34 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-09-11 10:15:00 | 2501.90 | 2490.72 | 2499.36 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 09:15:00 | 2591.20 | 2479.58 | 2479.17 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-10-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 12:15:00 | 2408.75 | 2480.49 | 2480.61 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 15:15:00 | 2558.35 | 2479.86 | 2479.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 09:15:00 | 2565.45 | 2480.71 | 2480.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 12:15:00 | 2500.00 | 2500.08 | 2490.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-13 10:15:00 | 2521.00 | 2500.43 | 2490.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 2857.65 | 2954.08 | 2853.97 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-29 09:15:00 | 2875.80 | 2951.52 | 2854.17 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 12:15:00 | 2887.25 | 2949.23 | 2854.46 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-30 09:15:00 | 2938.25 | 2947.61 | 2855.52 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-05 10:15:00 | 4212.60 | 4344.39 | 4223.40 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 14:15:00 | 4030.05 | 4580.10 | 4582.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 15:15:00 | 4015.85 | 4574.49 | 4579.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 4230.15 | 4204.18 | 4342.86 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 11:15:00 | 4658.95 | 4386.90 | 4385.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 12:15:00 | 4668.00 | 4389.70 | 4387.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 4452.90 | 4459.03 | 4426.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-03 11:15:00 | 4478.60 | 4459.23 | 4426.74 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 09:15:00 | 4411.85 | 4459.20 | 4427.53 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 4050.95 | 4398.79 | 4400.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 4015.05 | 4394.97 | 4398.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 4262.95 | 4238.83 | 4302.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-30 13:15:00 | 4200.05 | 4244.54 | 4300.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-31 11:15:00 | 4303.50 | 4245.16 | 4299.55 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 09:15:00 | 4559.00 | 4327.99 | 4327.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 4572.00 | 4369.27 | 4350.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 5172.50 | 5209.75 | 4992.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 5438.50 | 5201.18 | 4997.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5275.00 | 5423.30 | 5266.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 10:15:00 | 5305.50 | 5422.12 | 5266.39 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-13 11:15:00 | 5243.00 | 5420.34 | 5266.27 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 5598.00 | 5728.79 | 5729.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 09:15:00 | 5546.50 | 5724.36 | 5727.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 5714.00 | 5701.00 | 5713.80 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 5891.00 | 5724.69 | 5724.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 5905.00 | 5726.48 | 5725.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 10:15:00 | 5740.50 | 5764.86 | 5746.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-27 09:15:00 | 5810.00 | 5764.16 | 5746.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-28 13:15:00 | 5750.00 | 5769.70 | 5750.22 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 15:15:00 | 5604.50 | 5736.71 | 5736.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 5564.50 | 5735.00 | 5735.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 13:15:00 | 5734.00 | 5725.90 | 5731.20 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 5925.00 | 5736.41 | 5736.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 12:15:00 | 5932.00 | 5738.36 | 5737.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 12:15:00 | 5752.50 | 5761.49 | 5749.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-18 13:15:00 | 5778.00 | 5761.66 | 5749.79 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-18 14:15:00 | 5740.50 | 5761.45 | 5749.74 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 10:15:00 | 5298.50 | 5751.87 | 5752.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 5158.00 | 5726.99 | 5739.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 4914.80 | 4897.43 | 5105.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-19 14:15:00 | 4813.00 | 4927.05 | 5048.69 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 4641.70 | 4350.73 | 4577.67 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-09-08 14:15:00 | 2474.30 | 2023-09-11 10:15:00 | 2501.90 | EXIT_EMA400 | -27.60 |
| BUY | 2023-11-13 10:15:00 | 2521.00 | 2023-11-17 09:15:00 | 2611.16 | TARGET | 90.16 |
| BUY | 2024-01-29 09:15:00 | 2875.80 | 2024-01-30 09:15:00 | 2940.69 | TARGET | 64.89 |
| BUY | 2024-01-30 09:15:00 | 2938.25 | 2024-02-05 09:15:00 | 3186.44 | TARGET | 248.19 |
| BUY | 2025-01-03 11:15:00 | 4478.60 | 2025-01-06 09:15:00 | 4411.85 | EXIT_EMA400 | -66.75 |
| SELL | 2025-01-30 13:15:00 | 4200.05 | 2025-01-31 11:15:00 | 4303.50 | EXIT_EMA400 | -103.45 |
| BUY | 2025-05-12 09:15:00 | 5438.50 | 2025-06-13 11:15:00 | 5243.00 | EXIT_EMA400 | -195.50 |
| BUY | 2025-06-13 10:15:00 | 5305.50 | 2025-06-13 11:15:00 | 5243.00 | EXIT_EMA400 | -62.50 |
| BUY | 2025-10-27 09:15:00 | 5810.00 | 2025-10-28 13:15:00 | 5750.00 | EXIT_EMA400 | -60.00 |
| BUY | 2025-11-18 13:15:00 | 5778.00 | 2025-11-18 14:15:00 | 5740.50 | EXIT_EMA400 | -37.50 |
| SELL | 2026-02-19 14:15:00 | 4813.00 | 2026-03-09 09:15:00 | 4105.93 | TARGET | 707.07 |
