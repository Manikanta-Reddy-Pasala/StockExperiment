# Oil India Ltd. (OIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 490.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 30.46
- **Avg P&L per closed trade:** 6.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 536.05 | 586.44 | 586.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 14:15:00 | 523.40 | 585.33 | 585.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 520.25 | 510.79 | 534.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 10:15:00 | 508.45 | 511.17 | 533.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-03 09:15:00 | 487.95 | 457.75 | 484.60 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 423.90 | 400.60 | 400.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 426.85 | 400.86 | 400.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 441.60 | 444.30 | 429.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 09:15:00 | 448.50 | 441.76 | 430.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-11 09:15:00 | 431.15 | 442.44 | 433.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 400.05 | 433.82 | 433.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 11:15:00 | 397.10 | 421.94 | 427.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 406.50 | 405.18 | 413.95 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 431.50 | 415.75 | 415.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 434.60 | 416.09 | 415.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 425.65 | 428.04 | 423.29 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 404.15 | 420.57 | 420.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 404.00 | 420.40 | 420.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 412.05 | 411.30 | 415.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 10:15:00 | 402.50 | 411.03 | 414.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-31 09:15:00 | 418.65 | 410.42 | 413.98 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 448.60 | 416.76 | 416.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 465.00 | 417.54 | 417.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 453.20 | 463.49 | 446.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-17 12:15:00 | 465.25 | 463.03 | 447.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 448.35 | 462.92 | 447.87 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-19 09:15:00 | 464.70 | 462.24 | 448.04 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 460.95 | 472.88 | 460.92 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-16 11:15:00 | 459.30 | 472.74 | 460.92 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-26 10:15:00 | 508.45 | 2024-12-20 09:15:00 | 433.14 | TARGET | 75.31 |
| BUY | 2025-07-03 09:15:00 | 448.50 | 2025-07-11 09:15:00 | 431.15 | EXIT_EMA400 | -17.35 |
| SELL | 2025-12-26 10:15:00 | 402.50 | 2025-12-31 09:15:00 | 418.65 | EXIT_EMA400 | -16.15 |
| BUY | 2026-02-17 12:15:00 | 465.25 | 2026-03-16 11:15:00 | 459.30 | EXIT_EMA400 | -5.95 |
| BUY | 2026-02-19 09:15:00 | 464.70 | 2026-03-16 11:15:00 | 459.30 | EXIT_EMA400 | -5.40 |
