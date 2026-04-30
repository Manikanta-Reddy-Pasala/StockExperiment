# Cummins India Ltd. (CUMMINSIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 5266.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 555.50
- **Avg P&L per closed trade:** 92.58

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 10:15:00 | 1840.00 | 1742.35 | 1742.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 13:15:00 | 1851.85 | 1745.34 | 1743.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 1960.70 | 1977.88 | 1920.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-18 13:15:00 | 2028.20 | 1979.29 | 1922.67 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-04 11:15:00 | 3207.90 | 3514.43 | 3261.30 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 3580.80 | 3735.38 | 3735.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 3567.70 | 3729.20 | 3732.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 3596.90 | 3596.18 | 3653.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 09:15:00 | 3511.95 | 3594.23 | 3648.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-08 12:15:00 | 3675.00 | 3593.99 | 3647.54 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 3184.00 | 2932.28 | 2931.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 3274.20 | 2942.82 | 2937.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 09:15:00 | 3885.20 | 3941.11 | 3793.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 09:15:00 | 4010.70 | 3936.84 | 3822.56 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-05 13:15:00 | 4313.00 | 4427.57 | 4330.12 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 4017.70 | 4260.95 | 4261.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 3996.60 | 4236.41 | 4248.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 4205.10 | 4144.53 | 4192.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 14:15:00 | 4161.70 | 4148.11 | 4193.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 4161.70 | 4148.11 | 4193.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 11:15:00 | 4199.30 | 4149.37 | 4193.32 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 4425.80 | 4228.08 | 4227.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 10:15:00 | 4481.70 | 4234.56 | 4230.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 4590.20 | 4603.57 | 4471.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-12 10:15:00 | 4659.20 | 4604.12 | 4472.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 4523.30 | 4611.59 | 4487.55 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-16 14:15:00 | 4588.90 | 4611.37 | 4488.05 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-19 14:15:00 | 4495.00 | 4612.96 | 4501.24 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-18 13:15:00 | 2028.20 | 2024-02-05 09:15:00 | 2344.80 | TARGET | 316.60 |
| SELL | 2024-11-08 09:15:00 | 3511.95 | 2024-11-08 12:15:00 | 3675.00 | EXIT_EMA400 | -163.05 |
| BUY | 2025-10-10 09:15:00 | 4010.70 | 2025-12-11 09:15:00 | 4575.11 | TARGET | 564.41 |
| SELL | 2026-02-03 14:15:00 | 4161.70 | 2026-02-04 09:15:00 | 4066.06 | TARGET | 95.64 |
| BUY | 2026-03-12 10:15:00 | 4659.20 | 2026-03-19 14:15:00 | 4495.00 | EXIT_EMA400 | -164.20 |
| BUY | 2026-03-16 14:15:00 | 4588.90 | 2026-03-19 14:15:00 | 4495.00 | EXIT_EMA400 | -93.90 |
