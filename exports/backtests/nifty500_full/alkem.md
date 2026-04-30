# Alkem Laboratories Ltd. (ALKEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 5400.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 3 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 1 / 10
- **Target hits / EMA400 exits:** 1 / 10
- **Total realized P&L (per unit):** -986.58
- **Avg P&L per closed trade:** -89.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 11:15:00 | 3556.95 | 3675.75 | 3676.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 12:15:00 | 3525.10 | 3665.66 | 3670.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 09:15:00 | 3639.10 | 3610.44 | 3636.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-20 12:15:00 | 3552.25 | 3610.26 | 3632.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 3624.25 | 3600.41 | 3624.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-10-27 10:15:00 | 3645.25 | 3600.85 | 3624.60 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 14:15:00 | 3814.50 | 3644.75 | 3643.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 3836.00 | 3648.27 | 3645.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 4941.30 | 4967.63 | 4694.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-05 10:15:00 | 5074.95 | 4957.60 | 4769.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-02-26 12:15:00 | 4771.40 | 5190.09 | 4988.07 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 11:15:00 | 4830.95 | 4965.22 | 4965.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 4800.00 | 4963.58 | 4964.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 14:15:00 | 4876.70 | 4853.88 | 4901.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-02 09:15:00 | 4759.55 | 4861.94 | 4900.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-03 14:15:00 | 4902.40 | 4855.26 | 4894.72 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 15:15:00 | 5129.20 | 4928.47 | 4927.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 11:15:00 | 5181.40 | 4934.91 | 4931.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 4950.00 | 5150.06 | 5060.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-18 13:15:00 | 5146.95 | 5040.60 | 5021.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 5146.95 | 5040.60 | 5021.60 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-18 14:15:00 | 5214.40 | 5042.33 | 5022.57 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 5042.65 | 5067.22 | 5038.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-24 12:15:00 | 5012.00 | 5066.41 | 5038.04 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 5399.75 | 5889.84 | 5891.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 5356.50 | 5547.85 | 5649.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 09:15:00 | 5580.00 | 5538.45 | 5638.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-03 09:15:00 | 5495.90 | 5550.09 | 5632.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-07 10:15:00 | 5640.65 | 5545.93 | 5623.90 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 13:15:00 | 5093.00 | 4981.02 | 4980.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 14:15:00 | 5124.00 | 4982.44 | 4981.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 5098.50 | 5144.07 | 5079.52 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 4846.50 | 5035.58 | 5036.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 4790.50 | 5031.17 | 5033.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 4923.00 | 4920.66 | 4966.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-01 09:15:00 | 4884.10 | 4921.14 | 4965.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-15 14:15:00 | 4935.80 | 4874.68 | 4924.09 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 5037.90 | 4955.32 | 4955.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 5046.00 | 4956.22 | 4955.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 4932.00 | 4958.24 | 4956.52 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 4899.00 | 4954.58 | 4954.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 4835.50 | 4953.40 | 4954.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 4928.50 | 4917.17 | 4934.23 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 5389.50 | 4951.11 | 4950.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 5404.50 | 5010.86 | 4981.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 5404.50 | 5473.42 | 5374.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-29 10:15:00 | 5476.50 | 5468.69 | 5376.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 5584.00 | 5639.73 | 5556.80 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-09 10:15:00 | 5634.00 | 5639.67 | 5557.19 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 5582.50 | 5636.38 | 5573.03 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-18 12:15:00 | 5557.50 | 5634.62 | 5573.08 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 13:15:00 | 5379.00 | 5627.37 | 5628.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 5344.00 | 5561.25 | 5588.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 11:15:00 | 5450.50 | 5443.73 | 5515.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 09:15:00 | 5392.50 | 5443.12 | 5513.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 5440.50 | 5367.05 | 5450.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-13 09:15:00 | 5396.00 | 5368.01 | 5450.48 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-15 09:15:00 | 5463.00 | 5369.40 | 5448.34 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-20 12:15:00 | 3552.25 | 2023-10-27 10:15:00 | 3645.25 | EXIT_EMA400 | -93.00 |
| BUY | 2024-02-05 10:15:00 | 5074.95 | 2024-02-26 12:15:00 | 4771.40 | EXIT_EMA400 | -303.55 |
| SELL | 2024-05-02 09:15:00 | 4759.55 | 2024-05-03 14:15:00 | 4902.40 | EXIT_EMA400 | -142.85 |
| BUY | 2024-06-18 13:15:00 | 5146.95 | 2024-06-24 12:15:00 | 5012.00 | EXIT_EMA400 | -134.95 |
| BUY | 2024-06-18 14:15:00 | 5214.40 | 2024-06-24 12:15:00 | 5012.00 | EXIT_EMA400 | -202.40 |
| SELL | 2025-01-03 09:15:00 | 5495.90 | 2025-01-07 10:15:00 | 5640.65 | EXIT_EMA400 | -144.75 |
| SELL | 2025-07-01 09:15:00 | 4884.10 | 2025-07-15 14:15:00 | 4935.80 | EXIT_EMA400 | -51.70 |
| BUY | 2025-10-29 10:15:00 | 5476.50 | 2025-11-12 09:15:00 | 5777.12 | TARGET | 300.62 |
| BUY | 2025-12-09 10:15:00 | 5634.00 | 2025-12-18 12:15:00 | 5557.50 | EXIT_EMA400 | -76.50 |
| SELL | 2026-03-27 09:15:00 | 5392.50 | 2026-04-15 09:15:00 | 5463.00 | EXIT_EMA400 | -70.50 |
| SELL | 2026-04-13 09:15:00 | 5396.00 | 2026-04-15 09:15:00 | 5463.00 | EXIT_EMA400 | -67.00 |
