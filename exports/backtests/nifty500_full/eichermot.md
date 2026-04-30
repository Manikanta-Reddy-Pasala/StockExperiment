# Eicher Motors Ltd. (EICHERMOT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 7109.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 2 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 326.80
- **Avg P&L per closed trade:** 54.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 10:15:00 | 3492.25 | 3419.79 | 3419.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 12:15:00 | 3512.00 | 3421.46 | 3420.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 15:15:00 | 3446.35 | 3448.31 | 3435.96 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 12:15:00 | 3312.10 | 3424.72 | 3425.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 14:15:00 | 3296.25 | 3422.27 | 3423.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 10:15:00 | 3407.30 | 3406.18 | 3415.35 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2023-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 10:15:00 | 3517.65 | 3423.70 | 3423.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 11:15:00 | 3537.15 | 3424.83 | 3424.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 3896.10 | 3904.63 | 3755.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-21 09:15:00 | 3949.60 | 3905.08 | 3756.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-10 09:15:00 | 3816.90 | 3935.72 | 3832.77 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-03-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 14:15:00 | 3744.60 | 3827.99 | 3828.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 09:15:00 | 3702.20 | 3825.93 | 3827.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 09:15:00 | 3859.90 | 3811.78 | 3819.63 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 3998.25 | 3827.34 | 3827.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 11:15:00 | 4008.20 | 3850.16 | 3839.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 4541.00 | 4610.99 | 4420.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-06 14:15:00 | 4724.25 | 4606.71 | 4434.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-02 12:15:00 | 4731.30 | 4852.14 | 4742.67 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 4678.25 | 4805.25 | 4805.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 4567.35 | 4802.88 | 4804.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 4845.55 | 4790.91 | 4798.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-22 09:15:00 | 4772.95 | 4791.60 | 4798.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 4772.95 | 4791.60 | 4798.29 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-22 10:15:00 | 4799.90 | 4791.68 | 4798.29 | Close above EMA400 |

### Cycle 7 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 4912.25 | 4796.91 | 4796.40 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 4598.00 | 4796.17 | 4796.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 4566.60 | 4791.79 | 4794.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 4945.70 | 4784.97 | 4790.67 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 4893.10 | 4796.32 | 4796.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 4975.00 | 4800.50 | 4798.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 4821.35 | 4850.05 | 4826.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-06 09:15:00 | 4864.50 | 4840.23 | 4825.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 4827.50 | 4842.17 | 4827.11 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-09 13:15:00 | 4825.20 | 4842.00 | 4827.10 | Close below EMA400 |

### Cycle 10 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 4724.15 | 4816.63 | 4817.05 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 13:15:00 | 4879.75 | 4817.01 | 4816.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 4900.45 | 4821.54 | 4819.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 10:15:00 | 4943.40 | 4960.77 | 4899.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-14 10:15:00 | 5004.25 | 4960.47 | 4901.17 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-11 09:15:00 | 5055.50 | 5173.74 | 5060.21 | Close below EMA400 |

### Cycle 12 — SELL (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 09:15:00 | 4847.00 | 4986.61 | 4986.65 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 11:15:00 | 5106.85 | 4986.69 | 4986.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 5138.45 | 5005.60 | 4996.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 5104.50 | 5192.71 | 5112.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 5181.30 | 5185.69 | 5111.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-09 09:15:00 | 5302.50 | 5443.50 | 5319.58 | Close below EMA400 |

### Cycle 14 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 6854.00 | 7353.65 | 7355.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6745.00 | 7329.94 | 7343.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 11:15:00 | 7072.00 | 7049.56 | 7178.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-30 09:15:00 | 6984.00 | 7120.12 | 7172.37 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-21 09:15:00 | 3949.60 | 2024-01-10 09:15:00 | 3816.90 | EXIT_EMA400 | -132.70 |
| BUY | 2024-06-06 14:15:00 | 4724.25 | 2024-08-02 12:15:00 | 4731.30 | EXIT_EMA400 | 7.05 |
| SELL | 2024-10-22 09:15:00 | 4772.95 | 2024-10-22 10:15:00 | 4799.90 | EXIT_EMA400 | -26.95 |
| BUY | 2024-12-06 09:15:00 | 4864.50 | 2024-12-09 13:15:00 | 4825.20 | EXIT_EMA400 | -39.30 |
| BUY | 2025-01-14 10:15:00 | 5004.25 | 2025-02-03 09:15:00 | 5313.49 | TARGET | 309.24 |
| BUY | 2025-04-08 09:15:00 | 5181.30 | 2025-04-11 09:15:00 | 5390.76 | TARGET | 209.46 |
