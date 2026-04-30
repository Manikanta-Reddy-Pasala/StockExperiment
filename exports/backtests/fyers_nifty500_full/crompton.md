# Crompton Greaves Consumer Electricals Ltd. (CROMPTON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 270.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 26.85
- **Avg P&L per closed trade:** 6.71

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 407.65 | 435.53 | 435.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 403.65 | 433.17 | 434.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 11:15:00 | 401.35 | 399.01 | 410.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 15:15:00 | 396.55 | 405.53 | 409.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-12 10:15:00 | 355.50 | 342.47 | 354.57 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 348.30 | 345.21 | 345.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 350.00 | 345.37 | 345.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 347.45 | 347.47 | 346.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 13:15:00 | 349.00 | 347.04 | 346.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 349.00 | 347.04 | 346.29 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-16 15:15:00 | 349.55 | 347.08 | 346.32 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 347.25 | 347.29 | 346.47 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-18 13:15:00 | 346.05 | 347.27 | 346.46 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 334.45 | 346.86 | 346.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 330.95 | 344.78 | 345.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 329.85 | 329.39 | 335.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-20 09:15:00 | 327.30 | 329.38 | 335.50 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-29 10:15:00 | 333.55 | 327.38 | 333.20 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 277.85 | 250.24 | 250.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 278.80 | 250.53 | 250.37 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-17 15:15:00 | 396.55 | 2025-01-13 09:15:00 | 357.00 | TARGET | 39.55 |
| BUY | 2025-06-16 13:15:00 | 349.00 | 2025-06-18 13:15:00 | 346.05 | EXIT_EMA400 | -2.95 |
| BUY | 2025-06-16 15:15:00 | 349.55 | 2025-06-18 13:15:00 | 346.05 | EXIT_EMA400 | -3.50 |
| SELL | 2025-08-20 09:15:00 | 327.30 | 2025-08-29 10:15:00 | 333.55 | EXIT_EMA400 | -6.25 |
