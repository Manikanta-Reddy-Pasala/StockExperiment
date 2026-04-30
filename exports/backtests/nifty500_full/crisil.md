# CRISIL Ltd. (CRISIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 4288.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 7 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -612.55
- **Avg P&L per closed trade:** -87.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 3938.05 | 4146.93 | 4147.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 09:15:00 | 3887.75 | 4144.35 | 4146.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 14:15:00 | 4099.90 | 4085.17 | 4113.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-01 09:15:00 | 4062.15 | 4085.09 | 4113.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-02-01 14:15:00 | 4146.00 | 4084.35 | 4112.00 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-02-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 12:15:00 | 4539.00 | 4136.13 | 4134.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 11:15:00 | 4596.45 | 4240.74 | 4191.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 09:15:00 | 4871.50 | 4886.23 | 4698.80 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 11:15:00 | 4371.15 | 4642.54 | 4643.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 13:15:00 | 4343.10 | 4636.94 | 4640.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 4215.35 | 4214.00 | 4331.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-21 11:15:00 | 4137.45 | 4213.16 | 4329.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-07-01 09:15:00 | 4401.35 | 4207.98 | 4305.83 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 12:15:00 | 4549.75 | 4326.19 | 4325.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 4600.30 | 4364.57 | 4345.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 4581.10 | 4587.48 | 4509.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-30 09:15:00 | 4649.05 | 4584.28 | 4514.95 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-03 10:15:00 | 4497.30 | 4590.68 | 4523.32 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 4797.55 | 5432.47 | 5434.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 4596.05 | 5176.20 | 5290.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 14:15:00 | 4369.70 | 4368.05 | 4625.48 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 5018.70 | 4697.83 | 4696.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 5032.70 | 4704.38 | 4699.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 5779.50 | 5786.38 | 5551.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-24 12:15:00 | 5803.50 | 5786.59 | 5554.05 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 14:15:00 | 5538.50 | 5777.97 | 5559.92 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 13:15:00 | 5302.50 | 5433.93 | 5434.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 5273.50 | 5418.47 | 5426.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 4786.00 | 4783.83 | 4955.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-07 15:15:00 | 4724.70 | 4846.63 | 4937.01 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-06 11:15:00 | 4546.40 | 4415.15 | 4543.53 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 4735.90 | 4610.64 | 4610.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 4757.20 | 4612.10 | 4611.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 11:15:00 | 4611.00 | 4629.71 | 4620.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-12 12:15:00 | 4670.90 | 4630.12 | 4621.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 4670.90 | 4630.12 | 4621.03 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-13 09:15:00 | 4365.80 | 4628.62 | 4620.46 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 13:15:00 | 4560.30 | 4612.69 | 4612.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 4512.90 | 4610.60 | 4611.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 4605.30 | 4602.29 | 4607.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-24 09:15:00 | 4514.00 | 4601.74 | 4606.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 4205.10 | 4075.14 | 4222.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 12:15:00 | 4291.90 | 4077.29 | 4223.17 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-01 09:15:00 | 4062.15 | 2024-02-01 14:15:00 | 4146.00 | EXIT_EMA400 | -83.85 |
| SELL | 2024-06-21 11:15:00 | 4137.45 | 2024-07-01 09:15:00 | 4401.35 | EXIT_EMA400 | -263.90 |
| BUY | 2024-09-30 09:15:00 | 4649.05 | 2024-10-03 10:15:00 | 4497.30 | EXIT_EMA400 | -151.75 |
| BUY | 2025-07-24 12:15:00 | 5803.50 | 2025-07-25 14:15:00 | 5538.50 | EXIT_EMA400 | -265.00 |
| SELL | 2025-11-07 15:15:00 | 4724.70 | 2026-01-06 11:15:00 | 4546.40 | EXIT_EMA400 | 178.30 |
| BUY | 2026-02-12 12:15:00 | 4670.90 | 2026-02-13 09:15:00 | 4365.80 | EXIT_EMA400 | -305.10 |
| SELL | 2026-02-24 09:15:00 | 4514.00 | 2026-03-02 09:15:00 | 4235.25 | TARGET | 278.75 |
