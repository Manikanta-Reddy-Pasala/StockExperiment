# NIFTY 50 (^NSEI)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5015 bars)
- **Last close:** 23997.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 5000 pts (index)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / EMA400 exits:** 0 / 10
- **Total realized P&L (per unit):** 907.05
- **Avg P&L per closed trade:** 90.71

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 13:15:00 | 19119.60 | 19513.87 | 19514.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 09:15:00 | 19103.35 | 19502.53 | 19508.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 14:15:00 | 19416.50 | 19410.69 | 19457.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-07 09:15:00 | 19366.20 | 19410.26 | 19456.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 13:15:00 | 19441.40 | 19410.16 | 19454.24 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-08 15:15:00 | 19436.45 | 19410.80 | 19454.12 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 19447.65 | 19411.17 | 19454.09 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-09 14:15:00 | 19393.10 | 19411.82 | 19453.36 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-13 09:15:00 | 19469.45 | 19411.05 | 19451.13 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 14:15:00 | 19690.65 | 19485.72 | 19484.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 09:15:00 | 19766.95 | 19490.56 | 19487.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 21391.35 | 21397.13 | 20892.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-18 10:15:00 | 21484.25 | 21397.99 | 20895.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 21765.05 | 22046.58 | 21760.02 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-20 12:15:00 | 21908.45 | 22040.62 | 21761.28 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-04-18 14:15:00 | 21965.65 | 22287.43 | 22041.52 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 15:15:00 | 24238.50 | 24811.38 | 24813.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 23972.30 | 24803.03 | 24809.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 24261.75 | 24213.78 | 24457.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 12:15:00 | 24179.65 | 24215.51 | 24454.89 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 10:15:00 | 24416.75 | 24201.26 | 24403.68 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 09:15:00 | 24190.65 | 23185.17 | 23185.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 10:15:00 | 24356.05 | 23445.67 | 23324.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 24644.15 | 24672.47 | 24296.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 11:15:00 | 24688.95 | 24672.45 | 24300.55 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 10:15:00 | 24868.50 | 25133.80 | 24888.28 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 24576.90 | 24786.66 | 24787.12 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 25013.10 | 24788.28 | 24787.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 25074.70 | 24813.31 | 24800.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 24981.50 | 25001.59 | 24913.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-06 12:15:00 | 25060.05 | 24925.84 | 24888.22 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 25878.05 | 26035.65 | 25871.30 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-08 15:15:00 | 25868.90 | 26033.99 | 25871.29 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 25063.40 | 25764.41 | 25765.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 25029.40 | 25737.07 | 25751.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 25671.00 | 25585.77 | 25667.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 11:15:00 | 25629.80 | 25608.81 | 25673.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 25629.80 | 25608.81 | 25673.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-05 13:15:00 | 25596.95 | 25608.97 | 25672.99 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-06 14:15:00 | 25698.50 | 25608.92 | 25670.45 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-07 09:15:00 | 19366.20 | 2023-11-13 09:15:00 | 19469.45 | EXIT_EMA400 | -103.25 |
| SELL | 2023-11-08 15:15:00 | 19436.45 | 2023-11-13 09:15:00 | 19469.45 | EXIT_EMA400 | -33.00 |
| SELL | 2023-11-09 14:15:00 | 19393.10 | 2023-11-13 09:15:00 | 19469.45 | EXIT_EMA400 | -76.35 |
| BUY | 2024-01-18 10:15:00 | 21484.25 | 2024-04-18 14:15:00 | 21965.65 | EXIT_EMA400 | 481.40 |
| BUY | 2024-03-20 12:15:00 | 21908.45 | 2024-04-18 14:15:00 | 21965.65 | EXIT_EMA400 | 57.20 |
| SELL | 2024-11-25 12:15:00 | 24179.65 | 2024-12-03 10:15:00 | 24416.75 | EXIT_EMA400 | -237.10 |
| BUY | 2025-06-13 11:15:00 | 24688.95 | 2025-07-25 10:15:00 | 24868.50 | EXIT_EMA400 | 179.55 |
| BUY | 2025-10-06 12:15:00 | 25060.05 | 2026-01-08 15:15:00 | 25868.90 | EXIT_EMA400 | 808.85 |
| SELL | 2026-02-05 11:15:00 | 25629.80 | 2026-02-06 14:15:00 | 25698.50 | EXIT_EMA400 | -68.70 |
| SELL | 2026-02-05 13:15:00 | 25596.95 | 2026-02-06 14:15:00 | 25698.50 | EXIT_EMA400 | -101.55 |
