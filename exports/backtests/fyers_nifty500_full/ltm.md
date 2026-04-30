# LTM Ltd. (LTM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4283.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 355.43
- **Avg P&L per closed trade:** 59.24

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 09:15:00 | 5623.90 | 6083.22 | 6084.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 11:15:00 | 5601.25 | 5868.03 | 5923.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 10:15:00 | 4530.00 | 4476.58 | 4821.89 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 5131.40 | 4858.04 | 4857.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 5140.00 | 4860.85 | 4858.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 5233.00 | 5276.11 | 5164.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-16 09:15:00 | 5293.00 | 5262.60 | 5168.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-17 15:15:00 | 5169.00 | 5261.83 | 5173.78 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 5107.50 | 5135.86 | 5135.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 5080.00 | 5134.69 | 5135.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 5147.00 | 5133.14 | 5134.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 14:15:00 | 5104.50 | 5134.05 | 5134.96 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 5104.50 | 5134.05 | 5134.96 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-18 10:15:00 | 5089.00 | 5133.12 | 5134.48 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-20 09:15:00 | 5133.50 | 5127.54 | 5131.49 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 13:15:00 | 5243.00 | 5135.77 | 5135.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 5315.50 | 5145.70 | 5140.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 5156.50 | 5160.42 | 5148.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-01 14:15:00 | 5200.00 | 5157.00 | 5147.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 5155.50 | 5163.06 | 5151.40 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-03 12:15:00 | 5147.00 | 5162.97 | 5151.47 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 09:15:00 | 5648.00 | 5926.75 | 5926.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 5584.00 | 5899.10 | 5912.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 15:15:00 | 4454.90 | 4453.33 | 4829.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 09:15:00 | 4432.30 | 4456.73 | 4816.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 4760.40 | 4497.26 | 4776.72 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-17 15:15:00 | 4700.00 | 4506.70 | 4775.95 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 4759.40 | 4527.13 | 4773.28 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-22 09:15:00 | 4646.20 | 4537.03 | 4772.22 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-07-16 09:15:00 | 5293.00 | 2025-07-17 15:15:00 | 5169.00 | EXIT_EMA400 | -124.00 |
| SELL | 2025-08-14 14:15:00 | 5104.50 | 2025-08-20 09:15:00 | 5133.50 | EXIT_EMA400 | -29.00 |
| SELL | 2025-08-18 10:15:00 | 5089.00 | 2025-08-20 09:15:00 | 5133.50 | EXIT_EMA400 | -44.50 |
| BUY | 2025-09-01 14:15:00 | 5200.00 | 2025-09-03 12:15:00 | 5147.00 | EXIT_EMA400 | -53.00 |
| SELL | 2026-04-17 15:15:00 | 4700.00 | 2026-04-24 09:15:00 | 4472.14 | TARGET | 227.86 |
| SELL | 2026-04-22 09:15:00 | 4646.20 | 2026-04-24 11:15:00 | 4268.13 | TARGET | 378.07 |
