# Hindustan Aeronautics Ltd. (HAL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4330.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 487.00
- **Avg P&L per closed trade:** 81.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 4675.00 | 4830.30 | 4831.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 4630.35 | 4828.31 | 4830.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 4821.50 | 4796.85 | 4813.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-05 14:15:00 | 4791.95 | 4805.18 | 4816.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 4791.95 | 4805.18 | 4816.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-06 09:15:00 | 4737.00 | 4804.34 | 4815.53 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 4584.15 | 4485.51 | 4588.22 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-16 09:15:00 | 4614.00 | 4490.36 | 4588.14 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 11:15:00 | 4697.45 | 4450.12 | 4448.90 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 14:15:00 | 4227.45 | 4451.91 | 4452.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 14:15:00 | 4215.60 | 4437.14 | 4444.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 4186.70 | 4166.46 | 4274.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-21 10:15:00 | 4069.05 | 4167.65 | 4270.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 11:15:00 | 3698.20 | 3507.30 | 3691.92 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 4286.95 | 3814.40 | 3812.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 4324.10 | 3824.13 | 3817.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 4863.40 | 4904.28 | 4664.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 14:15:00 | 4975.00 | 4905.27 | 4675.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 4793.00 | 4911.57 | 4792.07 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-17 15:15:00 | 4781.20 | 4910.27 | 4792.02 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 4544.00 | 4722.49 | 4722.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 4526.00 | 4720.53 | 4721.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 4508.00 | 4501.58 | 4570.77 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 4915.20 | 4619.90 | 4619.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 4972.00 | 4755.43 | 4707.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 11:15:00 | 4769.30 | 4781.15 | 4728.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-11 09:15:00 | 4861.00 | 4734.38 | 4716.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 4751.50 | 4749.43 | 4725.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-20 13:15:00 | 4730.00 | 4754.03 | 4732.34 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 12:15:00 | 4436.00 | 4713.19 | 4713.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 4369.10 | 4610.46 | 4654.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 4452.00 | 4441.00 | 4535.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 14:15:00 | 4422.00 | 4440.81 | 4533.68 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 4523.10 | 4424.96 | 4506.58 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 4339.00 | 4115.48 | 4114.49 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-05 14:15:00 | 4791.95 | 2024-09-06 09:15:00 | 4719.60 | TARGET | 72.35 |
| SELL | 2024-09-06 09:15:00 | 4737.00 | 2024-09-17 10:15:00 | 4501.40 | TARGET | 235.60 |
| SELL | 2025-01-21 10:15:00 | 4069.05 | 2025-02-17 09:15:00 | 3464.11 | TARGET | 604.94 |
| BUY | 2025-06-20 14:15:00 | 4975.00 | 2025-07-17 15:15:00 | 4781.20 | EXIT_EMA400 | -193.80 |
| BUY | 2025-11-11 09:15:00 | 4861.00 | 2025-11-20 13:15:00 | 4730.00 | EXIT_EMA400 | -131.00 |
| SELL | 2025-12-24 14:15:00 | 4422.00 | 2026-01-05 09:15:00 | 4523.10 | EXIT_EMA400 | -101.10 |
