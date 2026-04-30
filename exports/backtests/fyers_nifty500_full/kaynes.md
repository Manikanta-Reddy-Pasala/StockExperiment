# Kaynes Technology India Ltd. (KAYNES.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4045.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 2
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -871.40
- **Avg P&L per closed trade:** -290.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 14:15:00 | 4943.05 | 6196.75 | 6201.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 10:15:00 | 4858.25 | 6159.11 | 6182.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 12:15:00 | 4468.05 | 4453.58 | 4880.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 4238.00 | 4666.63 | 4856.20 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 4825.00 | 4651.38 | 4837.11 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-09 14:15:00 | 4843.00 | 4658.74 | 4834.46 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 5985.10 | 4971.86 | 4971.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 6052.00 | 5473.11 | 5281.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 14:15:00 | 5821.00 | 5842.49 | 5595.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 10:15:00 | 5900.50 | 5723.05 | 5621.92 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-21 09:15:00 | 5773.00 | 5946.08 | 5818.72 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 6239.00 | 6673.95 | 6675.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 15:15:00 | 6210.00 | 6660.99 | 6668.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 3836.00 | 3759.37 | 4274.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-02 11:15:00 | 3732.60 | 3862.29 | 4130.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3835.40 | 3666.90 | 3852.34 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 10:15:00 | 3871.50 | 3668.93 | 3852.43 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 4220.20 | 3956.73 | 3955.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 4248.90 | 3959.64 | 3957.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 3988.00 | 3989.47 | 3973.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-30 13:15:00 | 4020.00 | 3989.64 | 3973.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 4020.00 | 3989.64 | 3973.49 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-04-30 14:15:00 | 4040.00 | 3990.14 | 3973.82 | Buy entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-07 09:15:00 | 4238.00 | 2025-04-09 14:15:00 | 4843.00 | EXIT_EMA400 | -605.00 |
| BUY | 2025-06-24 10:15:00 | 5900.50 | 2025-07-21 09:15:00 | 5773.00 | EXIT_EMA400 | -127.50 |
| SELL | 2026-03-02 11:15:00 | 3732.60 | 2026-04-08 10:15:00 | 3871.50 | EXIT_EMA400 | -138.90 |
