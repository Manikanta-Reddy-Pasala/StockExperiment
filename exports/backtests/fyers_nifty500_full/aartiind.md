# Aarti Industries Ltd. (AARTIIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 508.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -66.46
- **Avg P&L per closed trade:** -13.29

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 14:15:00 | 621.70 | 683.06 | 683.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 618.70 | 675.64 | 679.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 12:15:00 | 425.80 | 425.31 | 455.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-14 12:15:00 | 405.30 | 424.27 | 453.38 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-21 09:15:00 | 453.70 | 427.67 | 450.84 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 09:15:00 | 440.85 | 414.91 | 414.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 448.20 | 415.24 | 414.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 464.65 | 466.01 | 450.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 09:15:00 | 468.00 | 459.78 | 451.09 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-08 10:15:00 | 456.55 | 467.69 | 458.07 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 433.35 | 452.93 | 452.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 429.80 | 452.69 | 452.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 13:15:00 | 450.00 | 449.29 | 450.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-30 14:15:00 | 445.30 | 449.25 | 450.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 388.20 | 381.25 | 389.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-03 11:15:00 | 389.40 | 381.41 | 389.32 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 447.85 | 374.88 | 374.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 454.90 | 376.39 | 375.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 417.15 | 429.09 | 410.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-10 14:15:00 | 427.90 | 425.21 | 411.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 413.90 | 426.94 | 413.49 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-16 11:15:00 | 424.35 | 426.81 | 413.56 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-19 13:15:00 | 414.35 | 426.34 | 414.75 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-14 12:15:00 | 405.30 | 2025-01-21 09:15:00 | 453.70 | EXIT_EMA400 | -48.40 |
| BUY | 2025-06-25 09:15:00 | 468.00 | 2025-07-08 10:15:00 | 456.55 | EXIT_EMA400 | -11.45 |
| SELL | 2025-07-30 14:15:00 | 445.30 | 2025-07-31 09:15:00 | 428.36 | TARGET | 16.94 |
| BUY | 2026-03-10 14:15:00 | 427.90 | 2026-03-19 13:15:00 | 414.35 | EXIT_EMA400 | -13.55 |
| BUY | 2026-03-16 11:15:00 | 424.35 | 2026-03-19 13:15:00 | 414.35 | EXIT_EMA400 | -10.00 |
