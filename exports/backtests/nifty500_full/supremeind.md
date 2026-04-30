# Supreme Industries Ltd. (SUPREMEIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 3622.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -1165.25
- **Avg P&L per closed trade:** -233.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 11:15:00 | 4052.90 | 4312.23 | 4312.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 09:15:00 | 4034.10 | 4278.11 | 4294.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 10:15:00 | 4081.60 | 4067.57 | 4150.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-05 09:15:00 | 3984.55 | 4081.62 | 4143.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 4023.00 | 3962.60 | 4045.53 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-27 14:15:00 | 4188.15 | 3967.30 | 4045.85 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 11:15:00 | 4218.30 | 4097.94 | 4097.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 13:15:00 | 4249.80 | 4111.92 | 4105.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 09:15:00 | 5802.75 | 5839.89 | 5509.25 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 5079.60 | 5435.63 | 5436.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 15:15:00 | 5060.35 | 5313.56 | 5343.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 12:15:00 | 4694.70 | 4677.26 | 4863.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-29 15:15:00 | 4605.00 | 4676.56 | 4860.27 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-09 10:15:00 | 4919.65 | 4701.29 | 4842.80 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 4134.20 | 3638.61 | 3636.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 12:15:00 | 4168.60 | 3643.88 | 3639.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 13:15:00 | 4270.40 | 4270.41 | 4089.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-04 09:15:00 | 4293.30 | 4270.59 | 4091.97 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-11 09:15:00 | 4107.00 | 4253.13 | 4111.48 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 4180.30 | 4323.45 | 4323.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 4163.70 | 4301.85 | 4312.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 3445.00 | 3410.75 | 3603.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-21 09:15:00 | 3317.40 | 3469.31 | 3570.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3526.00 | 3463.22 | 3554.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-02 13:15:00 | 3563.20 | 3472.03 | 3548.12 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 3802.20 | 3601.22 | 3600.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 10:15:00 | 3815.50 | 3603.35 | 3601.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 11:15:00 | 3787.50 | 3817.70 | 3735.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-10 09:15:00 | 3907.70 | 3818.06 | 3737.26 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 3692.80 | 3866.54 | 3788.11 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 12:15:00 | 3707.90 | 3755.64 | 3755.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 13:15:00 | 3676.70 | 3754.86 | 3755.39 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-05 09:15:00 | 3984.55 | 2024-03-27 14:15:00 | 4188.15 | EXIT_EMA400 | -203.60 |
| SELL | 2024-11-29 15:15:00 | 4605.00 | 2024-12-09 10:15:00 | 4919.65 | EXIT_EMA400 | -314.65 |
| BUY | 2025-07-04 09:15:00 | 4293.30 | 2025-07-11 09:15:00 | 4107.00 | EXIT_EMA400 | -186.30 |
| SELL | 2026-01-21 09:15:00 | 3317.40 | 2026-02-02 13:15:00 | 3563.20 | EXIT_EMA400 | -245.80 |
| BUY | 2026-03-10 09:15:00 | 3907.70 | 2026-03-23 09:15:00 | 3692.80 | EXIT_EMA400 | -214.90 |
