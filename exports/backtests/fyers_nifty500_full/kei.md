# KEI Industries Ltd. (KEI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4875.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 128.33
- **Avg P&L per closed trade:** 25.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 4171.00 | 4364.11 | 4364.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 12:15:00 | 4099.25 | 4337.32 | 4350.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 4322.80 | 4318.96 | 4340.24 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 4571.10 | 4360.26 | 4359.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 4597.75 | 4368.95 | 4363.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 4349.90 | 4384.19 | 4371.78 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 4126.70 | 4360.07 | 4360.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 4078.00 | 4354.70 | 4357.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 4019.05 | 4003.44 | 4118.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 10:15:00 | 3990.05 | 4003.31 | 4117.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-27 09:15:00 | 4277.00 | 4007.54 | 4112.90 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 15:15:00 | 4479.90 | 4189.42 | 4188.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 11:15:00 | 4495.20 | 4197.11 | 4191.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 4232.85 | 4293.44 | 4247.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-19 13:15:00 | 4295.55 | 4291.47 | 4247.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 4295.55 | 4291.47 | 4247.87 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-20 10:15:00 | 4240.00 | 4290.26 | 4248.12 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 10:15:00 | 4091.65 | 4245.57 | 4245.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 11:15:00 | 4058.65 | 4243.71 | 4244.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 4414.15 | 4203.52 | 4222.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-27 09:15:00 | 4072.90 | 4227.56 | 4234.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 3158.60 | 2967.87 | 3164.68 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-05 12:15:00 | 3175.60 | 2969.94 | 3164.74 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 3554.40 | 3266.96 | 3266.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 3563.90 | 3269.92 | 3267.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 3685.90 | 3691.49 | 3576.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-14 09:15:00 | 3735.80 | 3688.06 | 3582.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 3731.60 | 3820.75 | 3730.68 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-12 11:15:00 | 3729.40 | 3819.06 | 3730.73 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3907.10 | 4178.65 | 4179.84 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 4414.30 | 4176.62 | 4176.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 4439.70 | 4181.59 | 4178.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 4552.50 | 4666.93 | 4494.07 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4089.00 | 4388.96 | 4389.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 4015.50 | 4385.25 | 4388.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 4500.10 | 4308.83 | 4345.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 11:15:00 | 4415.90 | 4324.17 | 4351.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 4415.90 | 4324.17 | 4351.94 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-09 12:15:00 | 4422.70 | 4325.15 | 4352.29 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 14:15:00 | 4646.00 | 4376.85 | 4375.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 4779.30 | 4383.62 | 4379.22 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-25 10:15:00 | 3990.05 | 2024-11-27 09:15:00 | 4277.00 | EXIT_EMA400 | -286.95 |
| BUY | 2024-12-19 13:15:00 | 4295.55 | 2024-12-20 10:15:00 | 4240.00 | EXIT_EMA400 | -55.55 |
| SELL | 2025-01-27 09:15:00 | 4072.90 | 2025-02-11 12:15:00 | 3588.87 | TARGET | 484.03 |
| BUY | 2025-07-14 09:15:00 | 3735.80 | 2025-08-12 11:15:00 | 3729.40 | EXIT_EMA400 | -6.40 |
| SELL | 2026-04-09 11:15:00 | 4415.90 | 2026-04-09 12:15:00 | 4422.70 | EXIT_EMA400 | -6.80 |
