# Tata Consultancy Services Ltd. (TCS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 2473.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 7 |
| ENTRY1 | 8 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 1
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** 630.84
- **Avg P&L per closed trade:** 70.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 14:15:00 | 3357.95 | 3462.07 | 3462.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 14:15:00 | 3351.80 | 3455.32 | 3458.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 3442.00 | 3420.27 | 3438.27 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 10:15:00 | 3483.65 | 3453.04 | 3452.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 3507.90 | 3454.13 | 3453.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 09:15:00 | 3665.50 | 3688.40 | 3607.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-05 10:15:00 | 3724.00 | 3687.68 | 3610.71 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 3977.90 | 4074.07 | 3973.06 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-19 15:15:00 | 3971.00 | 4073.04 | 3973.05 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 10:15:00 | 3856.40 | 3939.58 | 3939.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 11:15:00 | 3848.00 | 3938.66 | 3939.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 09:15:00 | 3923.00 | 3913.82 | 3925.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-16 12:15:00 | 3850.95 | 3917.65 | 3925.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-07 09:15:00 | 3896.70 | 3836.13 | 3873.80 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 4020.00 | 3881.21 | 3880.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 4038.10 | 3907.86 | 3895.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 4132.10 | 4175.88 | 4066.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-06 09:15:00 | 4229.65 | 4175.24 | 4069.54 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 4348.80 | 4434.09 | 4322.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-19 10:15:00 | 4312.15 | 4430.95 | 4322.67 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 4104.00 | 4278.09 | 4278.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 4084.65 | 4246.17 | 4261.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 4139.00 | 4136.25 | 4192.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 09:15:00 | 4101.35 | 4135.97 | 4191.13 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 4101.35 | 4135.97 | 4191.13 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-11 09:15:00 | 4199.05 | 4137.00 | 4187.92 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 14:15:00 | 4352.55 | 4204.69 | 4204.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 4382.60 | 4207.83 | 4206.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 4304.30 | 4309.41 | 4266.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-18 11:15:00 | 4330.05 | 4309.61 | 4266.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 4283.70 | 4310.83 | 4268.73 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-20 10:15:00 | 4265.15 | 4309.21 | 4268.96 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 15:15:00 | 4112.45 | 4239.05 | 4239.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 4099.70 | 4226.29 | 4232.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 09:15:00 | 4198.60 | 4185.89 | 4210.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-17 09:15:00 | 4141.15 | 4203.41 | 4215.90 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-12 12:15:00 | 3602.10 | 3452.77 | 3563.49 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 3160.90 | 3082.17 | 3082.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 10:15:00 | 3190.40 | 3097.86 | 3090.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 3204.30 | 3212.34 | 3168.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-01 10:15:00 | 3221.00 | 3212.38 | 3168.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 3191.80 | 3219.80 | 3182.70 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-12 12:15:00 | 3213.20 | 3219.54 | 3182.94 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 3190.90 | 3221.17 | 3186.63 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-14 15:15:00 | 3197.00 | 3220.93 | 3186.68 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 3208.50 | 3220.80 | 3186.79 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-19 09:15:00 | 3184.90 | 3219.59 | 3187.34 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 2991.80 | 3168.76 | 3168.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2934.00 | 3161.24 | 3165.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.00 | 2525.99 | 2688.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-10 09:15:00 | 2510.60 | 2531.89 | 2677.18 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-05 10:15:00 | 3724.00 | 2024-02-06 09:15:00 | 4063.86 | TARGET | 339.86 |
| SELL | 2024-05-16 12:15:00 | 3850.95 | 2024-06-04 12:15:00 | 3627.61 | TARGET | 223.34 |
| BUY | 2024-08-06 09:15:00 | 4229.65 | 2024-09-19 10:15:00 | 4312.15 | EXIT_EMA400 | 82.50 |
| SELL | 2024-11-07 09:15:00 | 4101.35 | 2024-11-11 09:15:00 | 4199.05 | EXIT_EMA400 | -97.70 |
| BUY | 2024-12-18 11:15:00 | 4330.05 | 2024-12-20 10:15:00 | 4265.15 | EXIT_EMA400 | -64.90 |
| SELL | 2025-01-17 09:15:00 | 4141.15 | 2025-02-13 14:15:00 | 3916.90 | TARGET | 224.25 |
| BUY | 2026-01-01 10:15:00 | 3221.00 | 2026-01-19 09:15:00 | 3184.90 | EXIT_EMA400 | -36.10 |
| BUY | 2026-01-12 12:15:00 | 3213.20 | 2026-01-19 09:15:00 | 3184.90 | EXIT_EMA400 | -28.30 |
| BUY | 2026-01-14 15:15:00 | 3197.00 | 2026-01-19 09:15:00 | 3184.90 | EXIT_EMA400 | -12.10 |
