# Trent Ltd. (TRENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4148.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** 70.13
- **Avg P&L per closed trade:** 11.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 6450.05 | 6978.60 | 6981.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 15:15:00 | 6420.00 | 6957.98 | 6970.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 6891.15 | 6866.09 | 6912.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-10 12:15:00 | 6820.85 | 6878.29 | 6914.36 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-11 12:15:00 | 6926.00 | 6878.35 | 6913.14 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 7049.10 | 6938.69 | 6938.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 7177.40 | 6949.16 | 6943.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 14:15:00 | 6932.05 | 6963.79 | 6951.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-31 13:15:00 | 7118.65 | 6965.21 | 6952.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 6992.65 | 7021.30 | 6984.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-06 14:15:00 | 6977.65 | 7020.86 | 6984.35 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 6547.95 | 6952.20 | 6952.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 6505.90 | 6947.76 | 6950.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 6199.95 | 6173.19 | 6471.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-04 09:15:00 | 5806.50 | 6166.79 | 6453.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 5488.50 | 5226.15 | 5509.41 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-28 11:15:00 | 5426.80 | 5228.14 | 5509.00 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-01 09:15:00 | 5564.00 | 5235.34 | 5505.70 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 5661.00 | 5388.31 | 5387.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 5796.00 | 5459.52 | 5426.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 5541.50 | 5551.36 | 5481.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 15:15:00 | 5610.00 | 5552.11 | 5484.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 5758.00 | 5843.24 | 5682.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-04 11:15:00 | 5573.00 | 5839.03 | 5682.36 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 5363.50 | 5581.54 | 5582.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 5345.00 | 5576.95 | 5579.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 5400.50 | 5362.00 | 5451.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-07 12:15:00 | 5311.00 | 5362.14 | 5447.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 5436.00 | 5358.15 | 5440.52 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-11 13:15:00 | 5477.00 | 5360.58 | 5440.51 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 4330.60 | 3908.61 | 3907.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 13:15:00 | 4362.00 | 3913.12 | 3909.38 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-10 12:15:00 | 6820.85 | 2024-12-11 12:15:00 | 6926.00 | EXIT_EMA400 | -105.15 |
| BUY | 2024-12-31 13:15:00 | 7118.65 | 2025-01-06 14:15:00 | 6977.65 | EXIT_EMA400 | -141.00 |
| SELL | 2025-02-04 09:15:00 | 5806.50 | 2025-04-01 09:15:00 | 5564.00 | EXIT_EMA400 | 242.50 |
| SELL | 2025-03-28 11:15:00 | 5426.80 | 2025-04-01 09:15:00 | 5564.00 | EXIT_EMA400 | -137.20 |
| BUY | 2025-06-13 15:15:00 | 5610.00 | 2025-06-20 15:15:00 | 5986.98 | TARGET | 376.98 |
| SELL | 2025-08-07 12:15:00 | 5311.00 | 2025-08-11 13:15:00 | 5477.00 | EXIT_EMA400 | -166.00 |
