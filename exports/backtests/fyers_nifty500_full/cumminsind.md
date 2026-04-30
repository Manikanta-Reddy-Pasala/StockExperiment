# Cummins India Ltd. (CUMMINSIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 5269.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 253.17
- **Avg P&L per closed trade:** 50.63

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 3580.40 | 3757.49 | 3757.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 3567.70 | 3729.36 | 3742.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 3597.10 | 3594.88 | 3658.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 15:15:00 | 3538.25 | 3593.68 | 3654.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-08 12:15:00 | 3675.00 | 3592.66 | 3652.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 3184.00 | 2932.25 | 2931.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 3274.20 | 2942.77 | 2936.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 09:15:00 | 3885.20 | 3941.11 | 3792.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 09:15:00 | 4010.70 | 3936.75 | 3822.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-05 13:15:00 | 4313.00 | 4427.16 | 4329.77 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 4017.70 | 4260.57 | 4261.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 15:15:00 | 4002.10 | 4238.13 | 4249.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 4205.10 | 4139.85 | 4188.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 14:15:00 | 4161.70 | 4143.65 | 4189.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 4161.70 | 4143.65 | 4189.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 11:15:00 | 4199.30 | 4145.08 | 4189.30 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 4425.60 | 4225.27 | 4224.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 10:15:00 | 4481.70 | 4231.94 | 4227.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 4590.20 | 4603.40 | 4470.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-12 10:15:00 | 4659.20 | 4603.95 | 4471.48 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 4523.30 | 4611.53 | 4486.36 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-16 14:15:00 | 4588.90 | 4611.30 | 4486.87 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-19 14:15:00 | 4495.00 | 4612.76 | 4500.09 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 15:15:00 | 3538.25 | 2024-11-08 12:15:00 | 3675.00 | EXIT_EMA400 | -136.75 |
| BUY | 2025-10-10 09:15:00 | 4010.70 | 2025-12-11 09:15:00 | 4575.36 | TARGET | 564.66 |
| SELL | 2026-02-03 14:15:00 | 4161.70 | 2026-02-04 09:15:00 | 4078.34 | TARGET | 83.36 |
| BUY | 2026-03-12 10:15:00 | 4659.20 | 2026-03-19 14:15:00 | 4495.00 | EXIT_EMA400 | -164.20 |
| BUY | 2026-03-16 14:15:00 | 4588.90 | 2026-03-19 14:15:00 | 4495.00 | EXIT_EMA400 | -93.90 |
