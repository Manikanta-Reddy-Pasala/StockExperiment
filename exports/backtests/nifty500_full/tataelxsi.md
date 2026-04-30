# Tata Elxsi Ltd. (TATAELXSI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 4129.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** 1113.43
- **Avg P&L per closed trade:** 123.71

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 14:15:00 | 7611.75 | 7337.81 | 7336.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 09:15:00 | 7701.60 | 7344.11 | 7339.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 14:15:00 | 7316.40 | 7387.76 | 7363.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-25 11:15:00 | 7505.55 | 7390.54 | 7365.25 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-18 09:15:00 | 8174.00 | 8634.45 | 8416.82 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 15:15:00 | 7644.60 | 8264.48 | 8267.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 7605.95 | 8191.01 | 8229.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 09:15:00 | 7885.00 | 7855.21 | 8006.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-23 11:15:00 | 7862.85 | 7855.44 | 8005.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 7838.30 | 7803.95 | 7955.08 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-03-01 13:15:00 | 7765.35 | 7804.94 | 7952.59 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-03-28 12:15:00 | 7839.10 | 7700.88 | 7819.50 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 8440.30 | 7108.10 | 7101.75 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 09:15:00 | 7039.45 | 7452.47 | 7453.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 7006.75 | 7427.03 | 7440.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 7102.45 | 6883.89 | 7073.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-01 10:15:00 | 6734.70 | 7030.63 | 7094.85 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 5520.50 | 5260.30 | 5578.22 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-23 12:15:00 | 5600.50 | 5269.64 | 5578.18 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 6234.00 | 5720.63 | 5719.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 6283.00 | 5830.95 | 5778.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 6295.00 | 6296.66 | 6112.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 12:15:00 | 6333.00 | 6294.01 | 6123.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 6162.00 | 6291.35 | 6153.82 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-02 09:15:00 | 6148.00 | 6286.49 | 6154.76 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 5828.00 | 6135.46 | 6136.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 5802.00 | 6132.15 | 6134.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 5733.50 | 5650.71 | 5815.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 09:15:00 | 5546.50 | 5674.73 | 5786.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 5591.50 | 5465.39 | 5582.89 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 10:15:00 | 5661.00 | 5346.10 | 5345.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 11:15:00 | 5758.50 | 5350.20 | 5347.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 12:15:00 | 5409.00 | 5426.16 | 5390.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-22 09:15:00 | 5464.00 | 5420.30 | 5389.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5464.00 | 5420.30 | 5389.05 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-23 13:15:00 | 5380.00 | 5421.02 | 5391.09 | Close below EMA400 |

### Cycle 8 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 5097.50 | 5374.43 | 5375.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 5025.00 | 5370.95 | 5373.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 4421.10 | 4379.38 | 4645.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 09:15:00 | 4378.10 | 4382.10 | 4637.47 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 4531.70 | 4401.77 | 4614.46 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 11:15:00 | 4524.00 | 4404.43 | 4613.68 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-17 09:15:00 | 4665.00 | 4413.60 | 4613.15 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-25 11:15:00 | 7505.55 | 2023-11-07 09:15:00 | 7926.45 | TARGET | 420.90 |
| SELL | 2024-02-23 11:15:00 | 7862.85 | 2024-03-13 14:15:00 | 7435.11 | TARGET | 427.74 |
| SELL | 2024-03-01 13:15:00 | 7765.35 | 2024-03-28 12:15:00 | 7839.10 | EXIT_EMA400 | -73.75 |
| SELL | 2025-01-01 10:15:00 | 6734.70 | 2025-02-27 09:15:00 | 5654.26 | TARGET | 1080.44 |
| BUY | 2025-06-23 12:15:00 | 6333.00 | 2025-07-02 09:15:00 | 6148.00 | EXIT_EMA400 | -185.00 |
| SELL | 2025-09-23 09:15:00 | 5546.50 | 2025-10-27 09:15:00 | 5591.50 | EXIT_EMA400 | -45.00 |
| BUY | 2026-01-22 09:15:00 | 5464.00 | 2026-01-23 13:15:00 | 5380.00 | EXIT_EMA400 | -84.00 |
| SELL | 2026-04-09 09:15:00 | 4378.10 | 2026-04-17 09:15:00 | 4665.00 | EXIT_EMA400 | -286.90 |
| SELL | 2026-04-16 11:15:00 | 4524.00 | 2026-04-17 09:15:00 | 4665.00 | EXIT_EMA400 | -141.00 |
