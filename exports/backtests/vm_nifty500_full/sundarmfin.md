# Sundaram Finance Ltd. (SUNDARMFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 4534.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -1105.40
- **Avg P&L per closed trade:** -122.82

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 13:15:00 | 4311.60 | 4468.86 | 4469.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 12:15:00 | 4264.00 | 4459.16 | 4464.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 09:15:00 | 4215.00 | 4168.90 | 4288.54 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 14:15:00 | 4970.00 | 4362.74 | 4360.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 5004.30 | 4375.09 | 4366.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 5004.95 | 5024.88 | 4831.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-18 14:15:00 | 5175.15 | 5000.92 | 4836.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 4979.80 | 5003.27 | 4850.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-24 09:15:00 | 4785.10 | 4997.46 | 4853.18 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 14:15:00 | 4165.85 | 4781.72 | 4783.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 15:15:00 | 4125.60 | 4697.51 | 4739.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 10:15:00 | 4381.50 | 4359.11 | 4517.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 13:15:00 | 4338.90 | 4358.96 | 4514.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 4467.90 | 4352.12 | 4484.24 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-18 11:15:00 | 4526.15 | 4361.07 | 4482.99 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 4607.10 | 4492.66 | 4492.16 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 4417.90 | 4491.66 | 4491.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 15:15:00 | 4391.75 | 4479.20 | 4485.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 4575.05 | 4480.15 | 4485.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 4393.55 | 4485.23 | 4488.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 4393.55 | 4485.23 | 4488.29 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-03 12:15:00 | 4358.50 | 4482.12 | 4486.68 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-04 09:15:00 | 4618.10 | 4478.93 | 4484.97 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 4689.75 | 4491.17 | 4490.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 4745.50 | 4509.95 | 4502.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 11:15:00 | 4523.15 | 4544.00 | 4521.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-27 14:15:00 | 4573.80 | 4535.61 | 4518.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 4573.80 | 4535.61 | 4518.95 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-28 09:15:00 | 4474.50 | 4535.47 | 4519.05 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 4645.50 | 4997.89 | 4998.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 4577.80 | 4990.11 | 4994.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 11:15:00 | 4805.90 | 4768.19 | 4862.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 09:15:00 | 4718.60 | 4869.09 | 4898.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 4669.40 | 4630.04 | 4725.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-23 11:15:00 | 4619.50 | 4630.20 | 4724.72 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 4624.50 | 4506.57 | 4597.52 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 15:15:00 | 4744.50 | 4644.21 | 4643.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 09:15:00 | 4769.60 | 4645.46 | 4644.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 4626.00 | 4658.11 | 4651.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-25 15:15:00 | 4764.10 | 4670.03 | 4658.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 4670.80 | 4675.33 | 4661.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-28 09:15:00 | 4645.40 | 4675.03 | 4662.02 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 4624.50 | 5135.36 | 5137.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 4592.50 | 5075.31 | 5105.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 4922.00 | 4900.20 | 5002.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 09:15:00 | 4804.30 | 4899.42 | 4999.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-16 09:15:00 | 5040.00 | 4894.41 | 4983.82 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-18 14:15:00 | 5175.15 | 2024-10-24 09:15:00 | 4785.10 | EXIT_EMA400 | -390.05 |
| SELL | 2024-12-09 13:15:00 | 4338.90 | 2024-12-18 11:15:00 | 4526.15 | EXIT_EMA400 | -187.25 |
| SELL | 2025-02-03 09:15:00 | 4393.55 | 2025-02-04 09:15:00 | 4618.10 | EXIT_EMA400 | -224.55 |
| SELL | 2025-02-03 12:15:00 | 4358.50 | 2025-02-04 09:15:00 | 4618.10 | EXIT_EMA400 | -259.60 |
| BUY | 2025-02-27 14:15:00 | 4573.80 | 2025-02-28 09:15:00 | 4474.50 | EXIT_EMA400 | -99.30 |
| SELL | 2025-09-23 11:15:00 | 4619.50 | 2025-09-29 14:15:00 | 4303.85 | TARGET | 315.65 |
| SELL | 2025-08-28 09:15:00 | 4718.60 | 2025-10-23 09:15:00 | 4624.50 | EXIT_EMA400 | 94.10 |
| BUY | 2025-11-25 15:15:00 | 4764.10 | 2025-11-28 09:15:00 | 4645.40 | EXIT_EMA400 | -118.70 |
| SELL | 2026-04-09 09:15:00 | 4804.30 | 2026-04-16 09:15:00 | 5040.00 | EXIT_EMA400 | -235.70 |
