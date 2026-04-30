# Delhivery Ltd. (DELHIVERY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 467.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 86.69
- **Avg P&L per closed trade:** 10.84

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 14:15:00 | 437.15 | 406.06 | 405.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 15:15:00 | 439.20 | 406.39 | 406.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 10:15:00 | 415.40 | 415.79 | 412.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-12 14:15:00 | 418.05 | 415.01 | 412.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 414.75 | 415.04 | 412.29 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-16 09:15:00 | 423.50 | 415.13 | 412.35 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 423.85 | 422.79 | 417.90 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-03 13:15:00 | 413.65 | 422.62 | 417.91 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 14:15:00 | 396.35 | 414.89 | 414.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 392.50 | 414.49 | 414.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 11:15:00 | 356.45 | 351.13 | 368.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-13 09:15:00 | 327.75 | 350.66 | 360.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-17 09:15:00 | 276.25 | 259.91 | 276.12 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 303.90 | 285.85 | 285.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 306.70 | 287.61 | 286.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 11:15:00 | 350.00 | 350.42 | 332.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 12:15:00 | 354.15 | 350.46 | 332.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-26 14:15:00 | 444.85 | 463.92 | 446.40 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 430.35 | 451.15 | 451.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 428.30 | 450.73 | 450.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 413.10 | 411.54 | 422.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-09 13:15:00 | 404.60 | 412.38 | 421.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 411.90 | 403.11 | 412.47 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 415.35 | 403.31 | 412.48 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 09:15:00 | 430.00 | 419.23 | 419.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 437.00 | 421.28 | 420.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 12:15:00 | 426.50 | 427.30 | 423.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-02 13:15:00 | 428.30 | 427.31 | 423.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 428.30 | 427.31 | 423.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-02 15:15:00 | 429.00 | 427.33 | 424.00 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-04 09:15:00 | 419.55 | 427.25 | 423.97 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 398.70 | 421.66 | 421.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 394.65 | 421.39 | 421.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 12:15:00 | 421.25 | 419.53 | 420.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 09:15:00 | 414.20 | 419.56 | 420.60 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-20 09:15:00 | 423.65 | 419.26 | 420.41 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 436.30 | 421.09 | 421.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 445.60 | 421.47 | 421.26 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-12 14:15:00 | 418.05 | 2024-09-23 09:15:00 | 435.70 | TARGET | 17.65 |
| BUY | 2024-09-16 09:15:00 | 423.50 | 2024-10-03 13:15:00 | 413.65 | EXIT_EMA400 | -9.85 |
| SELL | 2025-01-13 09:15:00 | 327.75 | 2025-04-17 09:15:00 | 276.25 | EXIT_EMA400 | 51.50 |
| BUY | 2025-06-20 12:15:00 | 354.15 | 2025-07-09 12:15:00 | 419.93 | TARGET | 65.78 |
| SELL | 2026-01-09 13:15:00 | 404.60 | 2026-01-30 09:15:00 | 415.35 | EXIT_EMA400 | -10.75 |
| BUY | 2026-03-02 13:15:00 | 428.30 | 2026-03-04 09:15:00 | 419.55 | EXIT_EMA400 | -8.75 |
| BUY | 2026-03-02 15:15:00 | 429.00 | 2026-03-04 09:15:00 | 419.55 | EXIT_EMA400 | -9.45 |
| SELL | 2026-03-19 09:15:00 | 414.20 | 2026-03-20 09:15:00 | 423.65 | EXIT_EMA400 | -9.45 |
