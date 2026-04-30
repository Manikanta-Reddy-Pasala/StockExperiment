# KEI Industries Ltd. (KEI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 4857.50
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
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 139.27
- **Avg P&L per closed trade:** 27.85

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 4240.00 | 4348.20 | 4348.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 12:15:00 | 4098.20 | 4336.68 | 4342.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 4322.80 | 4318.39 | 4332.87 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 13:15:00 | 4596.50 | 4346.49 | 4346.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 4612.50 | 4370.82 | 4358.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 4349.90 | 4383.74 | 4365.57 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 14:15:00 | 4054.95 | 4348.65 | 4349.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 4050.00 | 4324.37 | 4336.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 4019.05 | 4004.02 | 4117.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 10:15:00 | 3990.05 | 4003.88 | 4116.52 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-27 09:15:00 | 4277.00 | 4008.03 | 4111.51 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 14:15:00 | 4486.15 | 4186.87 | 4185.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 12:15:00 | 4505.00 | 4200.53 | 4192.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 4234.05 | 4293.51 | 4247.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-19 13:15:00 | 4295.55 | 4291.50 | 4247.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 4295.55 | 4291.50 | 4247.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-20 10:15:00 | 4240.00 | 4290.29 | 4247.31 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 11:15:00 | 4058.65 | 4243.70 | 4244.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 4041.05 | 4214.35 | 4228.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 4413.90 | 4203.51 | 4222.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-27 09:15:00 | 4072.90 | 4227.81 | 4234.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 3158.60 | 2968.06 | 3165.73 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-05 12:15:00 | 3175.60 | 2970.13 | 3165.78 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 3554.40 | 3266.97 | 3266.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 3563.90 | 3269.93 | 3268.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 3685.90 | 3691.51 | 3576.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-14 09:15:00 | 3735.80 | 3688.07 | 3582.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 3731.60 | 3820.68 | 3730.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-12 11:15:00 | 3729.40 | 3819.00 | 3730.74 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3908.50 | 4179.14 | 4180.23 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 15:15:00 | 4410.00 | 4179.00 | 4178.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 4439.70 | 4191.89 | 4185.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 4552.50 | 4668.87 | 4496.97 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 4135.00 | 4393.10 | 4393.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 4087.00 | 4390.05 | 4391.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 4500.10 | 4309.70 | 4347.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 09:15:00 | 4435.70 | 4323.05 | 4352.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 4435.70 | 4323.05 | 4352.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-09 10:15:00 | 4431.00 | 4324.13 | 4353.34 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 14:15:00 | 4646.00 | 4377.40 | 4377.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 4779.30 | 4384.16 | 4380.62 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-25 10:15:00 | 3990.05 | 2024-11-27 09:15:00 | 4277.00 | EXIT_EMA400 | -286.95 |
| BUY | 2024-12-19 13:15:00 | 4295.55 | 2024-12-20 10:15:00 | 4240.00 | EXIT_EMA400 | -55.55 |
| SELL | 2025-01-27 09:15:00 | 4072.90 | 2025-02-11 12:15:00 | 3589.43 | TARGET | 483.47 |
| BUY | 2025-07-14 09:15:00 | 3735.80 | 2025-08-12 11:15:00 | 3729.40 | EXIT_EMA400 | -6.40 |
| SELL | 2026-04-09 09:15:00 | 4435.70 | 2026-04-09 10:15:00 | 4431.00 | EXIT_EMA400 | 4.70 |
