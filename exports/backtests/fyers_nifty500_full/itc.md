# ITC Ltd. (ITC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 315.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -23.09
- **Avg P&L per closed trade:** -3.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 478.50 | 492.70 | 492.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 09:15:00 | 476.80 | 491.42 | 492.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 480.95 | 479.78 | 484.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-28 10:15:00 | 475.40 | 479.74 | 484.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 472.65 | 471.86 | 477.48 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-23 13:15:00 | 471.80 | 471.86 | 477.45 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-24 10:15:00 | 478.80 | 472.01 | 477.41 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 434.55 | 423.92 | 423.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 435.15 | 424.03 | 423.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 426.00 | 427.64 | 425.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-23 09:15:00 | 434.90 | 427.57 | 426.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-28 09:15:00 | 421.65 | 429.38 | 427.11 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 417.95 | 425.26 | 425.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 417.15 | 424.64 | 424.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 11:15:00 | 419.90 | 418.57 | 420.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-10 10:15:00 | 417.20 | 418.59 | 420.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-15 09:15:00 | 420.60 | 418.47 | 420.54 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 425.05 | 410.07 | 410.05 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 405.35 | 410.15 | 410.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 403.95 | 409.38 | 409.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 409.40 | 409.22 | 409.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-17 10:15:00 | 407.10 | 409.20 | 409.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 407.10 | 409.20 | 409.64 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-17 12:15:00 | 406.70 | 409.16 | 409.61 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 405.10 | 409.06 | 409.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-19 09:15:00 | 403.85 | 408.81 | 409.41 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-23 14:15:00 | 407.10 | 403.73 | 405.57 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-28 10:15:00 | 475.40 | 2024-12-24 10:15:00 | 478.80 | EXIT_EMA400 | -3.40 |
| SELL | 2024-12-23 13:15:00 | 471.80 | 2024-12-24 10:15:00 | 478.80 | EXIT_EMA400 | -7.00 |
| BUY | 2025-05-23 09:15:00 | 434.90 | 2025-05-28 09:15:00 | 421.65 | EXIT_EMA400 | -13.25 |
| SELL | 2025-07-10 10:15:00 | 417.20 | 2025-07-15 09:15:00 | 420.60 | EXIT_EMA400 | -3.40 |
| SELL | 2025-11-17 10:15:00 | 407.10 | 2025-12-01 09:15:00 | 399.49 | TARGET | 7.61 |
| SELL | 2025-11-17 12:15:00 | 406.70 | 2025-12-23 14:15:00 | 407.10 | EXIT_EMA400 | -0.40 |
| SELL | 2025-11-19 09:15:00 | 403.85 | 2025-12-23 14:15:00 | 407.10 | EXIT_EMA400 | -3.25 |
