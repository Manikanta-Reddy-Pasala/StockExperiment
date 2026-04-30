# Sundaram Finance Ltd. (SUNDARMFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4523.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -855.53
- **Avg P&L per closed trade:** -106.94

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 4959.35 | 4387.22 | 4387.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 5020.00 | 4590.17 | 4503.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 5004.95 | 5025.48 | 4837.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-18 14:15:00 | 5181.55 | 5001.49 | 4842.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 4979.80 | 5002.46 | 4854.86 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-24 09:15:00 | 4784.95 | 4996.77 | 4857.06 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 13:15:00 | 4181.15 | 4786.32 | 4789.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 4166.85 | 4780.16 | 4785.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 10:15:00 | 4381.55 | 4358.20 | 4518.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 13:15:00 | 4338.90 | 4358.09 | 4515.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 4467.90 | 4351.76 | 4485.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-18 11:15:00 | 4526.15 | 4360.74 | 4483.96 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 4607.10 | 4492.99 | 4492.81 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 12:15:00 | 4419.90 | 4492.48 | 4492.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 14:15:00 | 4414.20 | 4490.97 | 4492.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 4575.05 | 4480.34 | 4486.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 10:15:00 | 4390.95 | 4487.19 | 4489.60 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-04 09:15:00 | 4618.10 | 4481.81 | 4486.78 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 4681.70 | 4491.85 | 4491.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 4743.85 | 4511.44 | 4503.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 11:15:00 | 4523.15 | 4544.83 | 4522.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-27 14:15:00 | 4573.80 | 4536.12 | 4519.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 4573.80 | 4536.12 | 4519.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-28 09:15:00 | 4474.50 | 4535.94 | 4519.86 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 4645.50 | 4997.40 | 4998.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 4577.80 | 4989.62 | 4994.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 11:15:00 | 4805.90 | 4768.16 | 4862.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 09:15:00 | 4718.20 | 4869.30 | 4899.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 4669.40 | 4630.39 | 4725.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-23 11:15:00 | 4619.50 | 4630.54 | 4724.96 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 4624.50 | 4506.70 | 4597.69 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 15:15:00 | 4741.00 | 4643.84 | 4643.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 09:15:00 | 4769.60 | 4645.10 | 4644.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 4626.00 | 4657.65 | 4650.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 10:15:00 | 4765.00 | 4671.19 | 4659.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 4670.80 | 4674.66 | 4661.30 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-28 09:15:00 | 4645.40 | 4674.32 | 4661.59 | Close below EMA400 |

### Cycle 8 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 4651.00 | 5139.89 | 5140.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 4465.50 | 5050.32 | 5093.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 4922.00 | 4899.74 | 5003.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 09:15:00 | 4804.30 | 4898.86 | 4999.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-16 09:15:00 | 5040.00 | 4894.08 | 4984.15 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-18 14:15:00 | 5181.55 | 2024-10-24 09:15:00 | 4784.95 | EXIT_EMA400 | -396.60 |
| SELL | 2024-12-09 13:15:00 | 4338.90 | 2024-12-18 11:15:00 | 4526.15 | EXIT_EMA400 | -187.25 |
| SELL | 2025-02-03 10:15:00 | 4390.95 | 2025-02-04 09:15:00 | 4618.10 | EXIT_EMA400 | -227.15 |
| BUY | 2025-02-27 14:15:00 | 4573.80 | 2025-02-28 09:15:00 | 4474.50 | EXIT_EMA400 | -99.30 |
| SELL | 2025-09-23 11:15:00 | 4619.50 | 2025-09-29 14:15:00 | 4303.13 | TARGET | 316.37 |
| SELL | 2025-08-28 09:15:00 | 4718.20 | 2025-10-23 09:15:00 | 4624.50 | EXIT_EMA400 | 93.70 |
| BUY | 2025-11-26 10:15:00 | 4765.00 | 2025-11-28 09:15:00 | 4645.40 | EXIT_EMA400 | -119.60 |
| SELL | 2026-04-09 09:15:00 | 4804.30 | 2026-04-16 09:15:00 | 5040.00 | EXIT_EMA400 | -235.70 |
