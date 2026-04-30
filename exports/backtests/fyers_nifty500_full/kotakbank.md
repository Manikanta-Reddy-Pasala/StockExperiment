# Kotak Mahindra Bank Ltd. (KOTAKBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 384.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 31.47
- **Avg P&L per closed trade:** 6.29

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 11:15:00 | 347.33 | 361.91 | 361.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 345.75 | 361.75 | 361.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 13:15:00 | 353.00 | 352.52 | 356.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-03 09:15:00 | 349.00 | 353.11 | 355.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-05 12:15:00 | 355.88 | 352.82 | 355.44 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 383.13 | 355.85 | 355.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 15:15:00 | 386.26 | 369.99 | 364.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 383.10 | 383.55 | 375.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-05 10:15:00 | 387.44 | 383.33 | 376.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 413.00 | 427.83 | 412.81 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-12 09:15:00 | 431.38 | 425.17 | 413.75 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-15 09:15:00 | 414.20 | 424.73 | 414.67 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 390.78 | 424.37 | 424.40 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 428.36 | 407.83 | 407.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 429.10 | 408.05 | 407.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 423.80 | 424.03 | 417.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-01 09:15:00 | 431.24 | 420.59 | 418.57 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 429.80 | 432.00 | 427.46 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-07 09:15:00 | 426.96 | 431.89 | 427.47 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 410.40 | 425.22 | 425.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 406.50 | 423.79 | 424.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-27 14:15:00 | 414.65 | 422.89 | 423.21 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-03 09:15:00 | 349.00 | 2024-12-05 12:15:00 | 355.88 | EXIT_EMA400 | -6.88 |
| BUY | 2025-03-05 10:15:00 | 387.44 | 2025-03-24 09:15:00 | 421.58 | TARGET | 34.14 |
| BUY | 2025-05-12 09:15:00 | 431.38 | 2025-05-15 09:15:00 | 414.20 | EXIT_EMA400 | -17.18 |
| BUY | 2025-12-01 09:15:00 | 431.24 | 2026-01-07 09:15:00 | 426.96 | EXIT_EMA400 | -4.28 |
| SELL | 2026-02-27 14:15:00 | 414.65 | 2026-03-09 09:15:00 | 388.98 | TARGET | 25.67 |
