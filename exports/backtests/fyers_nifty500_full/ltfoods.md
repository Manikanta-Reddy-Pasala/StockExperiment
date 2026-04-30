# LT Foods Ltd. (LTFOODS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 430.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / EMA400 exits:** 5 / 3
- **Total realized P&L (per unit):** 120.45
- **Avg P&L per closed trade:** 15.06

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 367.65 | 397.45 | 397.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 13:15:00 | 359.50 | 395.30 | 396.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 12:15:00 | 393.10 | 391.57 | 394.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 10:15:00 | 387.95 | 392.77 | 394.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-12 12:15:00 | 395.00 | 392.16 | 394.12 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 400.25 | 365.57 | 365.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 414.15 | 366.75 | 366.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 13:15:00 | 427.00 | 427.71 | 406.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 436.30 | 427.76 | 406.55 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-23 11:15:00 | 402.75 | 427.68 | 407.45 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 439.50 | 452.08 | 452.08 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 460.60 | 451.97 | 451.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 465.70 | 453.24 | 452.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 454.85 | 455.98 | 454.14 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 424.80 | 452.30 | 452.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 423.35 | 452.01 | 452.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 421.85 | 420.31 | 431.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-28 14:15:00 | 415.45 | 421.14 | 430.57 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 417.00 | 420.66 | 429.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-31 10:15:00 | 410.85 | 420.56 | 429.50 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 407.20 | 400.22 | 408.59 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-24 09:15:00 | 401.15 | 400.38 | 408.51 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 405.00 | 368.76 | 382.39 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 428.25 | 392.01 | 391.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 429.40 | 392.38 | 392.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 399.70 | 402.73 | 398.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 11:15:00 | 412.00 | 399.46 | 397.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 412.00 | 399.46 | 397.00 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-05 13:15:00 | 418.40 | 399.72 | 397.15 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-06 14:15:00 | 388.30 | 401.41 | 398.12 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 387.50 | 395.38 | 395.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 13:15:00 | 386.15 | 395.22 | 395.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 397.65 | 395.14 | 395.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 09:15:00 | 387.70 | 395.12 | 395.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 406.20 | 386.71 | 390.15 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 14:15:00 | 412.55 | 393.17 | 393.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 417.75 | 393.78 | 393.41 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-11 10:15:00 | 387.95 | 2025-02-12 12:15:00 | 395.00 | EXIT_EMA400 | -7.05 |
| BUY | 2025-06-20 09:15:00 | 436.30 | 2025-06-23 11:15:00 | 402.75 | EXIT_EMA400 | -33.55 |
| SELL | 2025-10-28 14:15:00 | 415.45 | 2025-12-09 09:15:00 | 370.08 | TARGET | 45.37 |
| SELL | 2025-12-24 09:15:00 | 401.15 | 2025-12-30 13:15:00 | 379.07 | TARGET | 22.08 |
| SELL | 2025-10-31 10:15:00 | 410.85 | 2026-01-13 09:15:00 | 354.89 | TARGET | 55.96 |
| BUY | 2026-03-05 11:15:00 | 412.00 | 2026-03-06 09:15:00 | 457.00 | TARGET | 45.00 |
| BUY | 2026-03-05 13:15:00 | 418.40 | 2026-03-06 14:15:00 | 388.30 | EXIT_EMA400 | -30.10 |
| SELL | 2026-03-19 09:15:00 | 387.70 | 2026-03-23 10:15:00 | 364.97 | TARGET | 22.73 |
