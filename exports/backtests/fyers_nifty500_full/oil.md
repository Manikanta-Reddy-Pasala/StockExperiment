# Oil India Ltd. (OIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 491.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 24.64
- **Avg P&L per closed trade:** 6.16

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 539.30 | 586.91 | 587.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 12:15:00 | 536.05 | 586.40 | 586.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 520.60 | 510.33 | 533.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 10:15:00 | 508.75 | 510.80 | 533.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-03 09:15:00 | 487.95 | 457.69 | 484.46 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 424.15 | 400.33 | 400.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 426.90 | 400.83 | 400.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 441.60 | 444.31 | 429.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 09:15:00 | 448.55 | 441.77 | 430.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-11 09:15:00 | 431.15 | 442.45 | 433.33 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 400.05 | 433.76 | 433.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 11:15:00 | 397.10 | 421.89 | 427.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 406.50 | 405.18 | 413.93 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 431.60 | 415.74 | 415.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 434.60 | 416.08 | 415.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 425.65 | 428.04 | 423.28 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 404.00 | 420.41 | 420.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 399.50 | 419.88 | 420.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 412.05 | 411.36 | 415.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 10:15:00 | 402.50 | 411.08 | 414.68 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-31 09:15:00 | 418.65 | 410.47 | 414.01 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 448.60 | 416.78 | 416.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 15:15:00 | 449.50 | 417.11 | 416.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 453.20 | 465.72 | 448.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-19 13:15:00 | 476.10 | 464.06 | 449.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-16 10:15:00 | 460.95 | 473.37 | 461.80 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-26 10:15:00 | 508.75 | 2024-12-19 09:15:00 | 435.41 | TARGET | 73.34 |
| BUY | 2025-07-03 09:15:00 | 448.55 | 2025-07-11 09:15:00 | 431.15 | EXIT_EMA400 | -17.40 |
| SELL | 2025-12-26 10:15:00 | 402.50 | 2025-12-31 09:15:00 | 418.65 | EXIT_EMA400 | -16.15 |
| BUY | 2026-02-19 13:15:00 | 476.10 | 2026-03-16 10:15:00 | 460.95 | EXIT_EMA400 | -15.15 |
