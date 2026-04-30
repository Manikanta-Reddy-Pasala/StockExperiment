# Avenue Supermarts Ltd. (DMART.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4614.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -170.99
- **Avg P&L per closed trade:** -42.75

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 09:15:00 | 4683.50 | 5014.15 | 5014.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 4587.00 | 4998.52 | 5006.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 4134.10 | 3662.66 | 3883.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-08 09:15:00 | 3783.00 | 3716.48 | 3890.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-01 12:15:00 | 3900.85 | 3635.19 | 3755.82 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 15:15:00 | 3988.00 | 3722.70 | 3721.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 09:15:00 | 4018.10 | 3741.04 | 3731.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 10:15:00 | 4137.30 | 4147.52 | 3999.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-19 09:15:00 | 4230.50 | 4094.01 | 4013.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 4051.10 | 4100.39 | 4037.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-29 12:15:00 | 4022.80 | 4098.42 | 4036.97 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 3950.00 | 4115.07 | 4115.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 3941.00 | 4108.57 | 4111.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 11:15:00 | 4206.10 | 4102.00 | 4108.45 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 4293.20 | 4116.12 | 4115.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 4316.10 | 4170.86 | 4147.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 4616.60 | 4620.19 | 4480.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-24 12:15:00 | 4729.40 | 4622.48 | 4488.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 4548.60 | 4615.09 | 4495.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 4490.20 | 4613.02 | 4495.98 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 4305.00 | 4429.38 | 4429.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 4294.80 | 4426.75 | 4428.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 3847.40 | 3838.18 | 3965.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 3808.80 | 3837.95 | 3963.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-05 09:15:00 | 3854.50 | 3757.11 | 3851.04 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 4263.90 | 3871.74 | 3870.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 10:15:00 | 4302.20 | 3883.53 | 3876.24 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-08 09:15:00 | 3783.00 | 2025-01-14 09:15:00 | 3461.39 | TARGET | 321.61 |
| BUY | 2025-05-19 09:15:00 | 4230.50 | 2025-05-29 12:15:00 | 4022.80 | EXIT_EMA400 | -207.70 |
| BUY | 2025-09-24 12:15:00 | 4729.40 | 2025-09-29 11:15:00 | 4490.20 | EXIT_EMA400 | -239.20 |
| SELL | 2026-01-08 10:15:00 | 3808.80 | 2026-02-05 09:15:00 | 3854.50 | EXIT_EMA400 | -45.70 |
