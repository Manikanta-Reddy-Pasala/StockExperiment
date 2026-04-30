# Pfizer Ltd. (PFIZER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 4710.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 9 |
| ENTRY2 | 1 |
| EXIT | 9 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / EMA400 exits:** 3 / 7
- **Total realized P&L (per unit):** 835.76
- **Avg P&L per closed trade:** 83.58

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 11:15:00 | 3848.00 | 3871.34 | 3871.45 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 10:15:00 | 3903.00 | 3871.73 | 3871.61 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-09-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 14:15:00 | 3830.65 | 3871.56 | 3871.59 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 13:15:00 | 3914.70 | 3870.96 | 3870.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-04 14:15:00 | 3919.35 | 3871.44 | 3871.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-25 09:15:00 | 3893.20 | 3927.01 | 3905.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-25 15:15:00 | 3965.00 | 3927.78 | 3906.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 3938.00 | 3928.78 | 3908.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-10-27 13:15:00 | 3940.00 | 3929.05 | 3908.63 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-10-30 09:15:00 | 3905.25 | 3928.97 | 3908.89 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 14:15:00 | 4180.00 | 4359.39 | 4359.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 4143.50 | 4348.14 | 4353.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 4260.00 | 4242.80 | 4289.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-03 13:15:00 | 4200.40 | 4244.82 | 4286.32 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-07 13:15:00 | 4287.45 | 4246.08 | 4284.14 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 4670.10 | 4307.78 | 4306.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 4752.30 | 4408.46 | 4363.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 4600.95 | 4679.76 | 4548.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-10 10:15:00 | 4909.50 | 4647.02 | 4574.14 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-20 14:15:00 | 5417.50 | 5895.51 | 5639.42 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 10:15:00 | 5152.60 | 5583.74 | 5585.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 11:15:00 | 5115.00 | 5579.07 | 5582.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 5344.95 | 5335.23 | 5420.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-05 12:15:00 | 5288.65 | 5334.99 | 5405.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 5212.05 | 5087.05 | 5224.09 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-30 14:15:00 | 5333.40 | 5089.50 | 5224.64 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 14:15:00 | 4953.70 | 4303.08 | 4300.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 5006.50 | 4357.97 | 4329.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 5591.00 | 5591.96 | 5289.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-18 11:15:00 | 5692.00 | 5263.38 | 5249.50 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-26 11:15:00 | 5316.00 | 5388.90 | 5322.45 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 5153.00 | 5275.79 | 5276.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 5126.00 | 5264.38 | 5270.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 5149.00 | 5141.74 | 5196.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-01 11:15:00 | 5125.50 | 5141.52 | 5195.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 5135.00 | 5141.66 | 5194.97 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-03 09:15:00 | 5214.00 | 5142.36 | 5194.80 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 5351.00 | 5228.69 | 5228.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 5426.00 | 5247.27 | 5238.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 5252.00 | 5259.25 | 5245.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-30 14:15:00 | 5325.50 | 5259.78 | 5246.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 5325.50 | 5259.78 | 5246.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-31 11:15:00 | 5237.00 | 5260.04 | 5247.23 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 5070.50 | 5235.53 | 5235.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 5064.00 | 5233.83 | 5234.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 15:15:00 | 5037.00 | 5036.93 | 5093.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 13:15:00 | 4977.00 | 5046.46 | 5085.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-10 09:15:00 | 5117.90 | 4751.81 | 4860.35 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 5100.60 | 4931.28 | 4931.12 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 10:15:00 | 4853.50 | 4932.65 | 4932.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 4737.50 | 4927.12 | 4929.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4818.90 | 4787.77 | 4841.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 13:15:00 | 4739.80 | 4796.88 | 4837.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 4820.70 | 4796.23 | 4836.38 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-10 13:15:00 | 4852.70 | 4797.74 | 4836.35 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-25 15:15:00 | 3965.00 | 2023-10-30 09:15:00 | 3905.25 | EXIT_EMA400 | -59.75 |
| BUY | 2023-10-27 13:15:00 | 3940.00 | 2023-10-30 09:15:00 | 3905.25 | EXIT_EMA400 | -34.75 |
| SELL | 2024-05-03 13:15:00 | 4200.40 | 2024-05-07 13:15:00 | 4287.45 | EXIT_EMA400 | -87.05 |
| BUY | 2024-07-10 10:15:00 | 4909.50 | 2024-08-06 10:15:00 | 5915.59 | TARGET | 1006.09 |
| SELL | 2024-12-05 12:15:00 | 5288.65 | 2024-12-20 12:15:00 | 4937.11 | TARGET | 351.54 |
| BUY | 2025-08-18 11:15:00 | 5692.00 | 2025-08-26 11:15:00 | 5316.00 | EXIT_EMA400 | -376.00 |
| SELL | 2025-10-01 11:15:00 | 5125.50 | 2025-10-03 09:15:00 | 5214.00 | EXIT_EMA400 | -88.50 |
| BUY | 2025-10-30 14:15:00 | 5325.50 | 2025-10-31 11:15:00 | 5237.00 | EXIT_EMA400 | -88.50 |
| SELL | 2025-12-29 13:15:00 | 4977.00 | 2026-01-20 09:15:00 | 4651.42 | TARGET | 325.58 |
| SELL | 2026-04-09 13:15:00 | 4739.80 | 2026-04-10 13:15:00 | 4852.70 | EXIT_EMA400 | -112.90 |
