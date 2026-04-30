# Trent Ltd. (TRENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 4144.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
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
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 849.33
- **Avg P&L per closed trade:** 141.56

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 13:15:00 | 6462.60 | 6971.31 | 6972.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 15:15:00 | 6420.00 | 6960.99 | 6967.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 6891.15 | 6867.59 | 6909.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-10 12:15:00 | 6820.85 | 6880.01 | 6912.27 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-11 12:15:00 | 6926.00 | 6879.97 | 6911.14 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 7116.85 | 6935.07 | 6934.89 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 6880.00 | 6934.51 | 6934.63 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 7008.40 | 6935.39 | 6935.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 11:15:00 | 7027.05 | 6936.30 | 6935.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 14:15:00 | 6931.65 | 6964.80 | 6950.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-31 13:15:00 | 7122.00 | 6966.00 | 6951.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 6991.50 | 7021.98 | 6983.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-06 14:15:00 | 6979.75 | 7021.56 | 6983.42 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 6505.75 | 6948.63 | 6950.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 6407.40 | 6935.90 | 6943.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 6185.00 | 6183.09 | 6480.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 12:15:00 | 6099.10 | 6181.68 | 6476.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 5488.50 | 5227.36 | 5513.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-28 11:15:00 | 5425.35 | 5229.33 | 5513.42 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-01 09:15:00 | 5564.45 | 5236.66 | 5510.10 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 5642.00 | 5391.28 | 5390.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 5679.00 | 5442.59 | 5418.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 5541.50 | 5551.49 | 5482.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 13:15:00 | 5572.50 | 5551.36 | 5484.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 5758.00 | 5843.14 | 5683.38 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-04 11:15:00 | 5572.00 | 5838.92 | 5682.85 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 5363.50 | 5581.56 | 5582.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 5345.00 | 5577.02 | 5580.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 5404.00 | 5361.93 | 5452.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-07 12:15:00 | 5311.00 | 5362.27 | 5447.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 5436.00 | 5358.22 | 5440.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-11 13:15:00 | 5473.50 | 5360.61 | 5440.74 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 10:15:00 | 4226.50 | 3912.22 | 3911.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 13:15:00 | 4282.20 | 3922.31 | 3916.24 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-10 12:15:00 | 6820.85 | 2024-12-11 12:15:00 | 6926.00 | EXIT_EMA400 | -105.15 |
| BUY | 2024-12-31 13:15:00 | 7122.00 | 2025-01-06 14:15:00 | 6979.75 | EXIT_EMA400 | -142.25 |
| SELL | 2025-02-03 12:15:00 | 6099.10 | 2025-02-18 11:15:00 | 4965.76 | TARGET | 1133.34 |
| SELL | 2025-03-28 11:15:00 | 5425.35 | 2025-04-01 09:15:00 | 5564.45 | EXIT_EMA400 | -139.10 |
| BUY | 2025-06-13 13:15:00 | 5572.50 | 2025-06-20 14:15:00 | 5837.49 | TARGET | 264.99 |
| SELL | 2025-08-07 12:15:00 | 5311.00 | 2025-08-11 13:15:00 | 5473.50 | EXIT_EMA400 | -162.50 |
