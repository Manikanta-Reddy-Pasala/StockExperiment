# Apollo Hospitals Enterprise Ltd. (APOLLOHOSP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 7636.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 4 |
| EXIT | 7 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / EMA400 exits:** 4 / 7
- **Total realized P&L (per unit):** 398.87
- **Avg P&L per closed trade:** 36.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 13:15:00 | 4870.25 | 4993.23 | 4993.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 09:15:00 | 4853.00 | 4977.00 | 4984.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 09:15:00 | 4950.00 | 4945.69 | 4967.10 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 5130.15 | 4982.50 | 4982.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 09:15:00 | 5152.85 | 4988.89 | 4985.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 09:15:00 | 4960.80 | 4994.43 | 4988.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-25 11:15:00 | 5090.00 | 4996.76 | 4990.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 5021.00 | 5014.76 | 5000.56 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-09-29 09:15:00 | 5090.05 | 5015.85 | 5001.24 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-10-04 11:15:00 | 4986.25 | 5029.82 | 5009.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2023-10-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 11:15:00 | 4736.15 | 5005.17 | 5005.86 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-11-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 09:15:00 | 5169.05 | 4997.96 | 4997.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 10:15:00 | 5205.70 | 5000.03 | 4998.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 5400.95 | 5417.84 | 5293.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-21 11:15:00 | 5497.90 | 5418.96 | 5296.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 6123.85 | 6402.16 | 6123.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-29 13:15:00 | 6120.70 | 6399.36 | 6123.58 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 11:15:00 | 5860.50 | 6169.21 | 6169.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 12:15:00 | 5841.10 | 6165.94 | 6168.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 14:15:00 | 5961.45 | 5950.10 | 6022.12 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 6259.55 | 6066.47 | 6066.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 6292.90 | 6068.72 | 6067.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 6098.00 | 6100.72 | 6085.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-03 09:15:00 | 6138.80 | 6100.83 | 6085.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 6138.80 | 6100.83 | 6085.84 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-03 10:15:00 | 6163.40 | 6101.45 | 6086.23 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-04 12:15:00 | 6795.00 | 6977.36 | 6805.51 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 6767.00 | 7070.06 | 7071.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 6741.20 | 7025.97 | 7047.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 11:15:00 | 6951.25 | 6927.18 | 6986.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 09:15:00 | 6876.55 | 6928.13 | 6985.11 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-21 09:15:00 | 6567.85 | 6351.46 | 6537.14 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 11:15:00 | 7036.50 | 6623.10 | 6621.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 7075.00 | 6635.70 | 6627.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 6855.50 | 6862.91 | 6771.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 6911.50 | 6853.90 | 6771.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 6911.50 | 6853.90 | 6771.63 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-12 10:15:00 | 6926.00 | 6854.62 | 6772.40 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 6880.50 | 6937.99 | 6857.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-02 09:15:00 | 7065.00 | 6939.25 | 6858.36 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-03 09:15:00 | 6838.00 | 6938.68 | 6860.87 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 7455.50 | 7636.15 | 7636.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 7426.00 | 7627.53 | 7632.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 7259.50 | 7163.12 | 7294.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 09:15:00 | 7017.00 | 7214.29 | 7289.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-06 11:15:00 | 7170.00 | 7057.96 | 7164.58 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 7660.00 | 7240.65 | 7239.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 7670.00 | 7305.46 | 7273.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7512.00 | 7544.08 | 7431.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-13 11:15:00 | 7598.00 | 7544.62 | 7433.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.92 | 7436.28 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-16 13:15:00 | 7425.50 | 7541.01 | 7436.41 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-25 11:15:00 | 5090.00 | 2023-10-04 11:15:00 | 4986.25 | EXIT_EMA400 | -103.75 |
| BUY | 2023-09-29 09:15:00 | 5090.05 | 2023-10-04 11:15:00 | 4986.25 | EXIT_EMA400 | -103.80 |
| BUY | 2023-12-21 11:15:00 | 5497.90 | 2024-01-19 15:15:00 | 6102.26 | TARGET | 604.36 |
| BUY | 2024-07-03 09:15:00 | 6138.80 | 2024-07-05 13:15:00 | 6297.68 | TARGET | 158.88 |
| BUY | 2024-07-03 10:15:00 | 6163.40 | 2024-07-10 11:15:00 | 6394.92 | TARGET | 231.52 |
| SELL | 2025-02-06 09:15:00 | 6876.55 | 2025-02-11 09:15:00 | 6550.88 | TARGET | 325.67 |
| BUY | 2025-05-12 09:15:00 | 6911.50 | 2025-06-03 09:15:00 | 6838.00 | EXIT_EMA400 | -73.50 |
| BUY | 2025-05-12 10:15:00 | 6926.00 | 2025-06-03 09:15:00 | 6838.00 | EXIT_EMA400 | -88.00 |
| BUY | 2025-06-02 09:15:00 | 7065.00 | 2025-06-03 09:15:00 | 6838.00 | EXIT_EMA400 | -227.00 |
| SELL | 2026-01-20 09:15:00 | 7017.00 | 2026-02-06 11:15:00 | 7170.00 | EXIT_EMA400 | -153.00 |
| BUY | 2026-03-13 11:15:00 | 7598.00 | 2026-03-16 13:15:00 | 7425.50 | EXIT_EMA400 | -172.50 |
