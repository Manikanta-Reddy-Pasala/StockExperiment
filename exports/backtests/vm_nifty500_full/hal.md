# Hindustan Aeronautics Ltd. (HAL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 4338.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 43.72
- **Avg P&L per closed trade:** 8.74

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 11:15:00 | 4683.00 | 4784.75 | 4785.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 09:15:00 | 4667.00 | 4780.00 | 4782.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 11:15:00 | 4511.00 | 4483.00 | 4589.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-11 10:15:00 | 4453.00 | 4482.62 | 4585.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-15 11:15:00 | 4584.15 | 4485.49 | 4579.78 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 09:15:00 | 4687.00 | 4445.58 | 4445.27 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 15:15:00 | 4230.00 | 4449.78 | 4450.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 13:15:00 | 4218.10 | 4439.46 | 4445.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 4186.70 | 4166.52 | 4273.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-21 10:15:00 | 4070.40 | 4167.69 | 4270.22 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 11:15:00 | 3699.60 | 3508.54 | 3695.28 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 4287.15 | 3819.65 | 3817.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 4324.10 | 3824.67 | 3819.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 4863.40 | 4904.37 | 4665.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 14:15:00 | 4975.00 | 4905.37 | 4676.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 4793.00 | 4911.71 | 4792.38 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-17 15:15:00 | 4783.00 | 4910.43 | 4792.34 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 4539.10 | 4722.40 | 4722.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 4526.00 | 4720.45 | 4721.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 4508.00 | 4501.61 | 4570.85 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 4915.40 | 4619.93 | 4619.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 4970.20 | 4755.44 | 4707.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 11:15:00 | 4769.30 | 4781.28 | 4728.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-11 09:15:00 | 4861.00 | 4734.31 | 4716.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 4751.50 | 4749.35 | 4725.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-20 13:15:00 | 4730.00 | 4753.84 | 4732.25 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 12:15:00 | 4436.00 | 4713.04 | 4713.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 4369.10 | 4610.38 | 4654.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 4452.00 | 4440.98 | 4535.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 14:15:00 | 4421.50 | 4440.79 | 4533.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 4523.10 | 4424.95 | 4506.56 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 4351.30 | 4115.19 | 4114.18 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-11 10:15:00 | 4453.00 | 2024-10-15 11:15:00 | 4584.15 | EXIT_EMA400 | -131.15 |
| SELL | 2025-01-21 10:15:00 | 4070.40 | 2025-02-17 09:15:00 | 3470.93 | TARGET | 599.47 |
| BUY | 2025-06-20 14:15:00 | 4975.00 | 2025-07-17 15:15:00 | 4783.00 | EXIT_EMA400 | -192.00 |
| BUY | 2025-11-11 09:15:00 | 4861.00 | 2025-11-20 13:15:00 | 4730.00 | EXIT_EMA400 | -131.00 |
| SELL | 2025-12-24 14:15:00 | 4421.50 | 2026-01-05 09:15:00 | 4523.10 | EXIT_EMA400 | -101.60 |
