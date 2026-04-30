# Atul Ltd. (ATUL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 6817.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 1 |
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| EXIT | 8 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -259.22
- **Avg P&L per closed trade:** -32.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 13:15:00 | 6774.60 | 6997.83 | 6998.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 14:15:00 | 6772.55 | 6995.59 | 6997.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 11:15:00 | 6681.20 | 6651.00 | 6795.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-06 12:15:00 | 6640.00 | 6650.89 | 6794.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-12-04 10:15:00 | 6783.35 | 6621.70 | 6710.04 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 10:15:00 | 7094.80 | 6764.13 | 6763.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 10:15:00 | 7116.60 | 6853.61 | 6813.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 10:15:00 | 6899.95 | 6944.78 | 6872.07 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 6208.10 | 6826.07 | 6828.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 15:15:00 | 6150.00 | 6691.79 | 6756.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 11:15:00 | 6158.90 | 6012.58 | 6181.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-15 10:15:00 | 5985.20 | 6030.62 | 6179.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-03 09:15:00 | 6214.35 | 5984.37 | 6100.43 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 11:15:00 | 6499.70 | 6030.88 | 6030.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 13:15:00 | 6505.85 | 6039.73 | 6035.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 09:15:00 | 7680.00 | 7820.94 | 7514.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-01 09:15:00 | 7854.90 | 7726.40 | 7542.43 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-07 10:15:00 | 7558.05 | 7751.87 | 7575.43 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 13:15:00 | 7250.25 | 7613.62 | 7614.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 14:15:00 | 7238.10 | 7609.89 | 7612.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 12:15:00 | 7470.35 | 7458.49 | 7518.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-12 09:15:00 | 7307.60 | 7461.63 | 7517.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-28 14:15:00 | 6051.00 | 5765.29 | 6022.61 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 14:15:00 | 7027.50 | 6021.46 | 6018.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 13:15:00 | 7128.00 | 6546.51 | 6343.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 12:15:00 | 6998.00 | 7006.98 | 6744.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-18 14:15:00 | 7055.00 | 7007.09 | 6746.93 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-18 13:15:00 | 6986.50 | 7337.36 | 7100.49 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 6673.00 | 6949.90 | 6950.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 6571.00 | 6940.78 | 6946.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 6480.00 | 6479.09 | 6619.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-15 12:15:00 | 6420.00 | 6478.61 | 6618.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-18 09:15:00 | 6227.00 | 5931.58 | 6099.73 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 6185.50 | 6022.51 | 6022.27 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 5821.00 | 6022.49 | 6022.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 5755.50 | 6019.83 | 6021.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 5974.00 | 5953.01 | 5984.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 10:15:00 | 5886.00 | 5989.07 | 6000.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 5963.00 | 5986.45 | 5998.61 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 6401.50 | 5990.58 | 6000.62 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 10:15:00 | 6222.00 | 6010.31 | 6010.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 6310.00 | 6017.63 | 6013.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 6383.50 | 6405.54 | 6268.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-04 15:15:00 | 6508.00 | 6403.23 | 6271.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 6278.50 | 6410.96 | 6284.83 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-06 12:15:00 | 6640.00 | 2023-12-04 10:15:00 | 6783.35 | EXIT_EMA400 | -143.35 |
| SELL | 2024-04-15 10:15:00 | 5985.20 | 2024-05-03 09:15:00 | 6214.35 | EXIT_EMA400 | -229.15 |
| BUY | 2024-10-01 09:15:00 | 7854.90 | 2024-10-07 10:15:00 | 7558.05 | EXIT_EMA400 | -296.85 |
| SELL | 2024-12-12 09:15:00 | 7307.60 | 2025-01-13 09:15:00 | 6678.55 | TARGET | 629.05 |
| BUY | 2025-06-18 14:15:00 | 7055.00 | 2025-07-18 13:15:00 | 6986.50 | EXIT_EMA400 | -68.50 |
| SELL | 2025-09-15 12:15:00 | 6420.00 | 2025-10-14 14:15:00 | 5825.41 | TARGET | 594.59 |
| SELL | 2026-02-02 10:15:00 | 5886.00 | 2026-02-03 09:15:00 | 6401.50 | EXIT_EMA400 | -515.50 |
| BUY | 2026-03-04 15:15:00 | 6508.00 | 2026-03-09 09:15:00 | 6278.50 | EXIT_EMA400 | -229.50 |
