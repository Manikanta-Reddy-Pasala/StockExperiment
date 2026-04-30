# Blue Dart Express Ltd. (BLUEDART.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 5465.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 5 |
| EXIT | 8 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 3 / 10
- **Target hits / EMA400 exits:** 3 / 10
- **Total realized P&L (per unit):** 333.55
- **Avg P&L per closed trade:** 25.66

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 6703.10 | 6601.76 | 6601.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 11:15:00 | 6745.75 | 6607.94 | 6604.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 12:15:00 | 6625.60 | 6652.07 | 6629.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-10 12:15:00 | 6709.65 | 6652.83 | 6630.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 6709.65 | 6652.83 | 6630.96 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-10-10 13:15:00 | 6710.85 | 6653.41 | 6631.36 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 14:15:00 | 6673.40 | 6674.36 | 6645.64 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-10-17 09:15:00 | 6725.55 | 6675.01 | 6646.25 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 6655.70 | 6680.88 | 6651.23 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-10-19 10:15:00 | 6648.55 | 6680.56 | 6651.22 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 13:15:00 | 6406.15 | 6629.04 | 6630.13 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 13:15:00 | 6750.00 | 6624.18 | 6623.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 14:15:00 | 6758.60 | 6625.52 | 6624.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 11:15:00 | 6686.00 | 6688.83 | 6659.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-23 09:15:00 | 6741.00 | 6689.06 | 6660.54 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-16 14:15:00 | 7103.15 | 7285.15 | 7137.98 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 13:15:00 | 6499.00 | 7037.65 | 7039.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 12:15:00 | 6482.85 | 6974.87 | 7006.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 14:15:00 | 5965.15 | 5962.71 | 6246.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-27 09:15:00 | 5898.00 | 5961.97 | 6243.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 6173.00 | 5975.74 | 6231.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-02 09:15:00 | 6126.85 | 5987.33 | 6228.49 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-04-19 09:15:00 | 6193.00 | 6018.52 | 6170.00 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 09:15:00 | 7178.05 | 6258.13 | 6257.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 09:15:00 | 7283.15 | 6321.61 | 6289.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 12:15:00 | 6948.15 | 6976.27 | 6728.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 12:15:00 | 7122.85 | 6977.43 | 6738.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 7715.35 | 8048.90 | 7662.78 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-23 13:15:00 | 7822.20 | 8046.64 | 7663.58 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-29 14:15:00 | 7690.00 | 8004.04 | 7692.56 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 7977.75 | 8147.91 | 8147.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 13:15:00 | 7966.85 | 8143.59 | 8145.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 14:15:00 | 7698.00 | 7665.86 | 7828.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-19 14:15:00 | 7484.40 | 7693.42 | 7801.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-25 10:15:00 | 6370.00 | 6021.36 | 6312.70 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 09:15:00 | 6650.00 | 6346.46 | 6345.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 11:15:00 | 6694.50 | 6352.82 | 6348.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 6656.00 | 6698.02 | 6563.19 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 6172.50 | 6503.66 | 6503.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 6144.50 | 6500.08 | 6502.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 6433.50 | 6424.53 | 6458.01 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 6845.00 | 6489.14 | 6489.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 6905.50 | 6575.90 | 6538.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 6666.50 | 6698.34 | 6620.34 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 5820.00 | 6552.60 | 6555.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 10:15:00 | 5794.00 | 6545.05 | 6551.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 5871.00 | 5846.63 | 6031.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-18 12:15:00 | 5839.50 | 5846.63 | 6029.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 5951.00 | 5808.32 | 5966.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-30 14:15:00 | 5693.50 | 5809.89 | 5963.18 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-29 09:15:00 | 6140.00 | 5637.78 | 5781.80 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 6378.00 | 5904.52 | 5903.57 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 5734.00 | 5928.21 | 5928.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 5720.00 | 5909.07 | 5918.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 5575.00 | 5553.44 | 5676.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-31 09:15:00 | 5513.00 | 5553.04 | 5676.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-01 09:15:00 | 5785.00 | 5553.87 | 5672.19 | Close above EMA400 |

### Cycle 13 — BUY (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 15:15:00 | 5815.00 | 5613.32 | 5612.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 5846.00 | 5625.08 | 5618.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 5630.50 | 5652.33 | 5633.71 | EMA200 retest candle locked |

### Cycle 14 — SELL (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 14:15:00 | 5460.50 | 5620.22 | 5620.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 15:15:00 | 5425.50 | 5618.28 | 5619.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 12:15:00 | 5164.50 | 5142.35 | 5305.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 14:15:00 | 5077.80 | 5141.62 | 5297.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-17 09:15:00 | 5379.30 | 5149.75 | 5290.04 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-10 12:15:00 | 6709.65 | 2023-10-19 10:15:00 | 6648.55 | EXIT_EMA400 | -61.10 |
| BUY | 2023-10-10 13:15:00 | 6710.85 | 2023-10-19 10:15:00 | 6648.55 | EXIT_EMA400 | -62.30 |
| BUY | 2023-10-17 09:15:00 | 6725.55 | 2023-10-19 10:15:00 | 6648.55 | EXIT_EMA400 | -77.00 |
| BUY | 2023-11-23 09:15:00 | 6741.00 | 2023-11-28 09:15:00 | 6982.37 | TARGET | 241.37 |
| SELL | 2024-03-27 09:15:00 | 5898.00 | 2024-04-19 09:15:00 | 6193.00 | EXIT_EMA400 | -295.00 |
| SELL | 2024-04-02 09:15:00 | 6126.85 | 2024-04-19 09:15:00 | 6193.00 | EXIT_EMA400 | -66.15 |
| BUY | 2024-06-05 12:15:00 | 7122.85 | 2024-07-01 09:15:00 | 8277.37 | TARGET | 1154.52 |
| BUY | 2024-07-23 13:15:00 | 7822.20 | 2024-07-29 14:15:00 | 7690.00 | EXIT_EMA400 | -132.20 |
| SELL | 2024-12-19 14:15:00 | 7484.40 | 2024-12-31 09:15:00 | 6532.49 | TARGET | 951.91 |
| SELL | 2025-09-18 12:15:00 | 5839.50 | 2025-10-29 09:15:00 | 6140.00 | EXIT_EMA400 | -300.50 |
| SELL | 2025-09-30 14:15:00 | 5693.50 | 2025-10-29 09:15:00 | 6140.00 | EXIT_EMA400 | -446.50 |
| SELL | 2025-12-31 09:15:00 | 5513.00 | 2026-01-01 09:15:00 | 5785.00 | EXIT_EMA400 | -272.00 |
| SELL | 2026-04-13 14:15:00 | 5077.80 | 2026-04-17 09:15:00 | 5379.30 | EXIT_EMA400 | -301.50 |
