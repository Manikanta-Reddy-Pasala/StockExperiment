# BANK NIFTY (^NSEBANK)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5015 bars)
- **Last close:** 54863.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 5000 pts (index)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 2 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 0 / 9
- **Total realized P&L (per unit):** -5727.25
- **Avg P&L per closed trade:** -636.36

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 11:15:00 | 44218.70 | 44615.94 | 44617.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 44055.45 | 44602.11 | 44610.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 44703.60 | 44572.36 | 44593.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-08-30 11:15:00 | 44624.05 | 44573.71 | 44594.20 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 11:15:00 | 44624.05 | 44573.71 | 44594.20 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-08-30 12:15:00 | 44614.25 | 44574.11 | 44594.30 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 14:15:00 | 45575.30 | 44601.88 | 44597.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 15:15:00 | 45618.95 | 44612.00 | 44602.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 11:15:00 | 44923.65 | 45026.78 | 44835.84 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2023-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 14:15:00 | 44366.30 | 44716.57 | 44716.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 44026.10 | 44706.27 | 44711.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 44639.70 | 44626.22 | 44668.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-12 12:15:00 | 44555.85 | 44625.56 | 44667.23 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-15 09:15:00 | 44249.45 | 43722.13 | 44034.59 | Close above EMA400 |

### Cycle 4 — BUY (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 12:15:00 | 47003.55 | 44147.61 | 44144.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 14:15:00 | 47016.00 | 44204.10 | 44172.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 46953.35 | 47330.24 | 46469.67 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 10:15:00 | 45894.95 | 46042.54 | 46043.11 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 14:15:00 | 46207.30 | 46044.43 | 46044.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 15:15:00 | 46266.00 | 46046.64 | 46045.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 10:15:00 | 46204.20 | 46344.57 | 46216.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-01 11:15:00 | 46911.00 | 46322.81 | 46213.93 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-15 09:15:00 | 46499.55 | 46803.05 | 46523.93 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 13:15:00 | 50212.55 | 51593.56 | 51594.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 50026.05 | 51421.72 | 51504.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 52119.45 | 51343.69 | 51457.78 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 51977.55 | 51557.81 | 51557.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 52118.95 | 51586.67 | 51571.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 52388.90 | 52502.54 | 52123.61 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 51109.80 | 51871.45 | 51871.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 50968.85 | 51804.73 | 51837.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 49737.75 | 49669.82 | 50403.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 11:15:00 | 49286.45 | 49667.74 | 50394.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 09:15:00 | 50431.25 | 49660.13 | 50324.02 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 14:15:00 | 51594.60 | 49765.58 | 49761.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 52255.20 | 50237.00 | 50036.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 56408.90 | 56582.19 | 55678.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-21 09:15:00 | 56720.60 | 56566.32 | 55701.26 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 56048.00 | 56619.01 | 55896.88 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-29 13:15:00 | 56244.55 | 56598.01 | 55900.56 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 55607.20 | 56553.36 | 55911.78 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 53825.50 | 55619.39 | 55627.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 14:15:00 | 53608.00 | 55499.85 | 55566.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 54895.50 | 54875.94 | 55168.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 09:15:00 | 54650.45 | 55074.66 | 55205.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-01 11:15:00 | 55290.40 | 54983.58 | 55142.33 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 56086.30 | 55274.17 | 55272.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 56535.75 | 55312.70 | 55292.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 58847.85 | 58894.26 | 58116.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-17 15:15:00 | 58961.50 | 58894.90 | 58128.15 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-21 10:15:00 | 58627.10 | 59438.23 | 58893.79 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 55950.70 | 59395.18 | 59402.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 55745.50 | 59358.86 | 59384.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 55181.05 | 54841.57 | 56558.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 54741.00 | 54956.96 | 56447.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-15 09:15:00 | 56503.10 | 54997.54 | 56416.78 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-08-30 11:15:00 | 44624.05 | 2023-08-30 12:15:00 | 44614.25 | EXIT_EMA400 | 9.80 |
| SELL | 2023-10-12 12:15:00 | 44555.85 | 2023-11-15 09:15:00 | 44249.45 | EXIT_EMA400 | 306.40 |
| BUY | 2024-03-01 11:15:00 | 46911.00 | 2024-03-15 09:15:00 | 46499.55 | EXIT_EMA400 | -411.45 |
| SELL | 2025-02-01 11:15:00 | 49286.45 | 2025-02-05 09:15:00 | 50431.25 | EXIT_EMA400 | -1144.80 |
| BUY | 2025-07-21 09:15:00 | 56720.60 | 2025-07-31 09:15:00 | 55607.20 | EXIT_EMA400 | -1113.40 |
| BUY | 2025-07-29 13:15:00 | 56244.55 | 2025-07-31 09:15:00 | 55607.20 | EXIT_EMA400 | -637.35 |
| SELL | 2025-09-26 09:15:00 | 54650.45 | 2025-10-01 11:15:00 | 55290.40 | EXIT_EMA400 | -639.95 |
| BUY | 2025-12-17 15:15:00 | 58961.50 | 2026-01-21 10:15:00 | 58627.10 | EXIT_EMA400 | -334.40 |
| SELL | 2026-04-13 09:15:00 | 54741.00 | 2026-04-15 09:15:00 | 56503.10 | EXIT_EMA400 | -1762.10 |
