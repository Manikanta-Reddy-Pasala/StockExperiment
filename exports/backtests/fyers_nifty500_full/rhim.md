# RHI MAGNESITA INDIA LTD. (RHIM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3408 bars)
- **Last close:** 404.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -90.00
- **Avg P&L per closed trade:** -12.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 15:15:00 | 488.00 | 462.19 | 462.19 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 13:15:00 | 445.00 | 462.58 | 462.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 442.65 | 462.38 | 462.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 12:15:00 | 457.15 | 455.25 | 458.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-13 12:15:00 | 449.30 | 454.98 | 458.23 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-14 09:15:00 | 464.15 | 454.90 | 458.13 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 471.40 | 460.41 | 460.38 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 456.05 | 460.34 | 460.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 445.65 | 460.13 | 460.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 456.70 | 456.67 | 458.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-09 13:15:00 | 448.25 | 456.47 | 458.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 456.60 | 456.25 | 458.06 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-10 13:15:00 | 452.95 | 456.20 | 458.01 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 456.90 | 456.18 | 457.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-11 11:15:00 | 452.60 | 456.13 | 457.93 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-06-11 14:15:00 | 460.80 | 456.14 | 457.91 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 13:15:00 | 510.00 | 459.86 | 459.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 10:15:00 | 521.80 | 461.77 | 460.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 10:15:00 | 476.00 | 476.30 | 469.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 13:15:00 | 482.50 | 475.23 | 470.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 470.65 | 475.39 | 470.47 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-07 15:15:00 | 466.80 | 475.31 | 470.46 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 470.75 | 484.50 | 484.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 467.50 | 481.60 | 482.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 10:15:00 | 493.05 | 466.94 | 474.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-09 09:15:00 | 447.70 | 467.77 | 474.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 460.60 | 457.40 | 465.65 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-29 13:15:00 | 465.70 | 457.64 | 465.61 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 483.65 | 469.78 | 469.77 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 454.35 | 469.79 | 469.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 453.00 | 466.66 | 468.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 458.40 | 453.62 | 459.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-24 12:15:00 | 446.75 | 453.32 | 459.16 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-31 10:15:00 | 459.60 | 452.25 | 457.88 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-05-13 12:15:00 | 449.30 | 2025-05-14 09:15:00 | 464.15 | EXIT_EMA400 | -14.85 |
| SELL | 2025-06-09 13:15:00 | 448.25 | 2025-06-11 14:15:00 | 460.80 | EXIT_EMA400 | -12.55 |
| SELL | 2025-06-10 13:15:00 | 452.95 | 2025-06-11 14:15:00 | 460.80 | EXIT_EMA400 | -7.85 |
| SELL | 2025-06-11 11:15:00 | 452.60 | 2025-06-11 14:15:00 | 460.80 | EXIT_EMA400 | -8.20 |
| BUY | 2025-07-03 13:15:00 | 482.50 | 2025-07-07 15:15:00 | 466.80 | EXIT_EMA400 | -15.70 |
| SELL | 2025-10-09 09:15:00 | 447.70 | 2025-10-29 13:15:00 | 465.70 | EXIT_EMA400 | -18.00 |
| SELL | 2025-12-24 12:15:00 | 446.75 | 2025-12-31 10:15:00 | 459.60 | EXIT_EMA400 | -12.85 |
