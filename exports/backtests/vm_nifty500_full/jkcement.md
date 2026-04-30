# J.K. Cement Ltd. (JKCEMENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 5287.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 8 |
| ENTRY1 | 9 |
| ENTRY2 | 4 |
| EXIT | 9 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 2 / 11
- **Target hits / EMA400 exits:** 1 / 12
- **Total realized P&L (per unit):** -522.95
- **Avg P&L per closed trade:** -40.23

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 13:15:00 | 3399.30 | 3244.49 | 3244.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 14:15:00 | 3413.25 | 3246.17 | 3245.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 13:15:00 | 3256.50 | 3261.64 | 3253.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-13 13:15:00 | 3284.20 | 3260.21 | 3253.10 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 3284.20 | 3260.21 | 3253.10 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-09-13 14:15:00 | 3294.80 | 3260.55 | 3253.31 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-09-20 09:15:00 | 3248.85 | 3270.63 | 3259.46 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 15:15:00 | 3144.15 | 3249.23 | 3249.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 12:15:00 | 3136.45 | 3221.21 | 3234.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 11:15:00 | 3239.95 | 3201.31 | 3221.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-11 14:15:00 | 3185.15 | 3201.18 | 3221.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 3216.75 | 3201.18 | 3221.18 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-10-12 10:15:00 | 3242.05 | 3201.59 | 3221.28 | Close above EMA400 |

### Cycle 3 — BUY (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 10:15:00 | 3409.05 | 3224.94 | 3224.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 3475.50 | 3261.75 | 3243.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 09:15:00 | 4115.65 | 4151.66 | 3972.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-16 09:15:00 | 4235.45 | 4151.56 | 3978.76 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 09:15:00 | 4133.65 | 4278.37 | 4138.46 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 10:15:00 | 3979.10 | 4140.08 | 4140.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 3952.15 | 4130.50 | 4135.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 4028.55 | 3999.14 | 4054.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-29 09:15:00 | 3968.95 | 3998.71 | 4048.36 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-03 09:15:00 | 4051.00 | 3983.13 | 4035.24 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 14:15:00 | 4331.35 | 4066.48 | 4065.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 09:15:00 | 4352.80 | 4071.73 | 4067.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 4208.95 | 4242.08 | 4171.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-04 14:15:00 | 4293.90 | 4243.25 | 4173.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 4284.60 | 4351.47 | 4276.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-05 09:15:00 | 4221.75 | 4350.18 | 4276.60 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 4228.85 | 4447.96 | 4448.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 4185.05 | 4436.04 | 4442.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 4284.45 | 4178.52 | 4274.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 09:15:00 | 4169.95 | 4180.57 | 4272.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 4221.80 | 4177.77 | 4260.67 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-29 14:15:00 | 4261.95 | 4179.69 | 4260.40 | Close above EMA400 |

### Cycle 7 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 4580.35 | 4326.38 | 4325.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 4599.05 | 4333.48 | 4328.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 4570.00 | 4593.89 | 4504.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-23 14:15:00 | 4789.80 | 4564.98 | 4511.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 4642.90 | 4731.85 | 4640.13 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-14 12:15:00 | 4628.50 | 4730.82 | 4640.07 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 11:15:00 | 4431.40 | 4591.80 | 4592.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 14:15:00 | 4425.75 | 4587.22 | 4589.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 4530.90 | 4511.68 | 4547.15 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 15:15:00 | 4743.90 | 4576.56 | 4576.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 10:15:00 | 4784.45 | 4590.16 | 4583.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 11:15:00 | 4690.25 | 4708.57 | 4650.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 13:15:00 | 4789.30 | 4709.62 | 4651.28 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 6644.50 | 6838.08 | 6558.79 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-08 10:15:00 | 6700.00 | 6836.71 | 6559.49 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 6575.00 | 6811.65 | 6571.63 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-11 09:15:00 | 6610.50 | 6809.65 | 6571.82 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-12 13:15:00 | 6561.50 | 6788.95 | 6573.92 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 6361.00 | 6539.34 | 6539.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 6279.50 | 6513.10 | 6525.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 14:15:00 | 5925.00 | 5888.60 | 6116.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-27 09:15:00 | 5823.00 | 5888.31 | 6114.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 5760.00 | 5640.06 | 5803.74 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-06 09:15:00 | 5717.00 | 5643.20 | 5802.88 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-07 09:15:00 | 5821.00 | 5648.05 | 5799.85 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-13 13:15:00 | 3284.20 | 2023-09-20 09:15:00 | 3248.85 | EXIT_EMA400 | -35.35 |
| BUY | 2023-09-13 14:15:00 | 3294.80 | 2023-09-20 09:15:00 | 3248.85 | EXIT_EMA400 | -45.95 |
| SELL | 2023-10-11 14:15:00 | 3185.15 | 2023-10-12 10:15:00 | 3242.05 | EXIT_EMA400 | -56.90 |
| BUY | 2024-02-16 09:15:00 | 4235.45 | 2024-03-13 09:15:00 | 4133.65 | EXIT_EMA400 | -101.80 |
| SELL | 2024-05-29 09:15:00 | 3968.95 | 2024-06-03 09:15:00 | 4051.00 | EXIT_EMA400 | -82.05 |
| BUY | 2024-07-04 14:15:00 | 4293.90 | 2024-08-05 09:15:00 | 4221.75 | EXIT_EMA400 | -72.15 |
| SELL | 2024-11-26 09:15:00 | 4169.95 | 2024-11-29 14:15:00 | 4261.95 | EXIT_EMA400 | -92.00 |
| BUY | 2025-01-23 14:15:00 | 4789.80 | 2025-02-14 12:15:00 | 4628.50 | EXIT_EMA400 | -161.30 |
| BUY | 2025-04-07 13:15:00 | 4789.30 | 2025-04-22 10:15:00 | 5203.35 | TARGET | 414.05 |
| BUY | 2025-09-08 10:15:00 | 6700.00 | 2025-09-12 13:15:00 | 6561.50 | EXIT_EMA400 | -138.50 |
| BUY | 2025-09-11 09:15:00 | 6610.50 | 2025-09-12 13:15:00 | 6561.50 | EXIT_EMA400 | -49.00 |
| SELL | 2025-11-27 09:15:00 | 5823.00 | 2026-01-07 09:15:00 | 5821.00 | EXIT_EMA400 | 2.00 |
| SELL | 2026-01-06 09:15:00 | 5717.00 | 2026-01-07 09:15:00 | 5821.00 | EXIT_EMA400 | -104.00 |
