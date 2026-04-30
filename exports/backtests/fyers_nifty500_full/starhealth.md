# Star Health and Allied Insurance Company Ltd. (STARHEALTH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 526.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 121.29
- **Avg P&L per closed trade:** 24.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 551.30 | 590.71 | 590.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 547.75 | 583.63 | 587.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 495.05 | 490.92 | 520.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 09:15:00 | 471.35 | 490.74 | 517.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-16 09:15:00 | 389.70 | 363.51 | 386.87 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 440.00 | 391.95 | 391.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 447.80 | 394.28 | 393.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 438.95 | 442.78 | 425.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 11:15:00 | 451.65 | 442.85 | 425.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 426.70 | 442.03 | 426.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 425.05 | 441.86 | 426.64 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 458.95 | 473.27 | 473.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 455.30 | 472.43 | 472.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 469.30 | 465.28 | 468.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-02 12:15:00 | 461.00 | 465.29 | 468.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 455.10 | 448.39 | 456.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 469.75 | 448.79 | 456.84 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 472.30 | 461.79 | 461.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 476.60 | 462.04 | 461.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 463.10 | 463.55 | 462.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-25 09:15:00 | 467.50 | 462.11 | 462.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 467.50 | 462.11 | 462.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-02 09:15:00 | 460.60 | 463.52 | 462.80 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 445.60 | 462.14 | 462.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 440.55 | 461.92 | 462.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 463.00 | 459.88 | 460.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-13 10:15:00 | 458.60 | 459.88 | 460.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 458.60 | 459.88 | 460.92 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-13 14:15:00 | 464.50 | 459.94 | 460.93 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 470.30 | 460.54 | 460.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 15:15:00 | 475.00 | 460.93 | 460.73 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-09 09:15:00 | 471.35 | 2025-04-07 09:15:00 | 333.94 | TARGET | 137.41 |
| BUY | 2025-06-16 11:15:00 | 451.65 | 2025-06-19 12:15:00 | 425.05 | EXIT_EMA400 | -26.60 |
| SELL | 2026-01-02 12:15:00 | 461.00 | 2026-01-19 10:15:00 | 437.72 | TARGET | 23.28 |
| BUY | 2026-02-25 09:15:00 | 467.50 | 2026-03-02 09:15:00 | 460.60 | EXIT_EMA400 | -6.90 |
| SELL | 2026-03-13 10:15:00 | 458.60 | 2026-03-13 14:15:00 | 464.50 | EXIT_EMA400 | -5.90 |
