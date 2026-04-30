# Britannia Industries Ltd. (BRITANNIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 5726.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -441.48
- **Avg P&L per closed trade:** -49.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 12:15:00 | 4728.00 | 4603.50 | 4602.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 4771.90 | 4630.94 | 4618.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 12:15:00 | 5059.65 | 5069.27 | 4927.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-10 14:15:00 | 5084.95 | 5069.42 | 4928.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-23 13:15:00 | 4961.00 | 5085.51 | 4972.55 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 14:15:00 | 4837.60 | 4975.33 | 4975.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 4833.80 | 4972.59 | 4974.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 10:15:00 | 4947.05 | 4945.14 | 4958.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-13 11:15:00 | 4894.70 | 4944.63 | 4958.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-03-15 09:15:00 | 4966.15 | 4939.39 | 4954.72 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 10:15:00 | 5067.30 | 4894.82 | 4894.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 09:15:00 | 5108.30 | 4904.93 | 4899.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 5669.20 | 5694.08 | 5512.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-05 09:15:00 | 5780.55 | 5695.93 | 5519.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 5952.35 | 6090.39 | 5950.86 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-14 10:15:00 | 5912.45 | 6088.62 | 5950.67 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 5625.20 | 5883.74 | 5884.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 5593.65 | 5876.30 | 5880.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 4928.45 | 4879.19 | 5088.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-15 09:15:00 | 4822.30 | 4889.42 | 5066.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 5025.15 | 4891.03 | 5031.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-24 09:15:00 | 5068.70 | 4896.54 | 5031.04 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 5343.60 | 4913.10 | 4912.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 5362.85 | 4930.01 | 4920.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 10:15:00 | 5520.00 | 5524.70 | 5407.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 11:15:00 | 5545.00 | 5524.90 | 5408.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 5614.50 | 5715.82 | 5612.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 12:15:00 | 5610.50 | 5714.77 | 5612.69 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 5412.50 | 5574.88 | 5575.51 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 5662.50 | 5576.54 | 5576.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 09:15:00 | 5690.00 | 5577.66 | 5576.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 5931.00 | 5948.59 | 5817.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-24 12:15:00 | 6015.00 | 5948.74 | 5820.63 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-08 09:15:00 | 5844.00 | 5953.40 | 5856.58 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 5816.50 | 5877.87 | 5877.91 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 5941.50 | 5877.98 | 5877.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 5960.00 | 5878.80 | 5878.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 5863.00 | 5879.86 | 5878.90 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 5849.00 | 5877.80 | 5877.88 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5955.00 | 5877.99 | 5877.97 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 5825.50 | 5877.53 | 5877.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 5820.00 | 5876.96 | 5877.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 5891.00 | 5875.03 | 5876.46 | EMA200 retest candle locked |

### Cycle 13 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 6010.00 | 5878.35 | 5878.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 6039.50 | 5881.47 | 5879.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 5955.00 | 5978.09 | 5940.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-05 09:15:00 | 5992.00 | 5978.32 | 5941.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 5992.00 | 5978.32 | 5941.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-05 10:15:00 | 6024.50 | 5978.77 | 5941.61 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 6002.00 | 6001.71 | 5957.24 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-08 13:15:00 | 6050.00 | 6002.59 | 5958.35 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 5964.50 | 6003.66 | 5960.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-09 13:15:00 | 5948.00 | 6003.11 | 5960.14 | Close below EMA400 |

### Cycle 14 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 5710.00 | 5935.13 | 5935.29 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 5995.50 | 5927.02 | 5926.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 10:15:00 | 6042.50 | 5928.17 | 5927.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 5993.00 | 6021.23 | 5982.23 | EMA200 retest candle locked |

### Cycle 16 — SELL (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 13:15:00 | 5806.50 | 5956.32 | 5957.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 5707.00 | 5939.72 | 5948.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 5682.00 | 5670.05 | 5770.21 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-10 14:15:00 | 5084.95 | 2024-01-23 13:15:00 | 4961.00 | EXIT_EMA400 | -123.95 |
| SELL | 2024-03-13 11:15:00 | 4894.70 | 2024-03-15 09:15:00 | 4966.15 | EXIT_EMA400 | -71.45 |
| BUY | 2024-08-05 09:15:00 | 5780.55 | 2024-10-14 10:15:00 | 5912.45 | EXIT_EMA400 | 131.90 |
| SELL | 2025-01-15 09:15:00 | 4822.30 | 2025-01-24 09:15:00 | 5068.70 | EXIT_EMA400 | -246.40 |
| BUY | 2025-06-20 11:15:00 | 5545.00 | 2025-07-25 12:15:00 | 5610.50 | EXIT_EMA400 | 65.50 |
| BUY | 2025-09-24 12:15:00 | 6015.00 | 2025-10-08 09:15:00 | 5844.00 | EXIT_EMA400 | -171.00 |
| BUY | 2026-01-05 09:15:00 | 5992.00 | 2026-01-07 09:15:00 | 6144.42 | TARGET | 152.42 |
| BUY | 2026-01-05 10:15:00 | 6024.50 | 2026-01-09 13:15:00 | 5948.00 | EXIT_EMA400 | -76.50 |
| BUY | 2026-01-08 13:15:00 | 6050.00 | 2026-01-09 13:15:00 | 5948.00 | EXIT_EMA400 | -102.00 |
