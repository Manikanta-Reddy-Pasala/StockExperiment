# Vardhman Textiles Ltd. (VTL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 613.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 1
- **Winners / losers:** 0 / 8
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -126.65
- **Avg P&L per closed trade:** -15.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 09:15:00 | 476.25 | 497.49 | 497.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 11:15:00 | 473.20 | 497.01 | 497.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 14:15:00 | 483.95 | 476.36 | 484.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 14:15:00 | 466.50 | 476.16 | 483.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 471.70 | 460.44 | 472.99 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-01 17:15:00 | 473.00 | 460.90 | 472.49 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 13:15:00 | 513.00 | 473.89 | 473.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 518.20 | 477.23 | 475.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 09:15:00 | 509.80 | 510.83 | 496.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-30 09:15:00 | 518.45 | 510.91 | 497.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 502.85 | 510.11 | 498.46 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-02 12:15:00 | 506.55 | 510.07 | 498.50 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-01-06 10:15:00 | 496.50 | 510.28 | 499.29 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 13:15:00 | 459.00 | 492.67 | 492.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 14:15:00 | 456.95 | 492.31 | 492.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 10:15:00 | 404.20 | 404.10 | 425.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-27 10:15:00 | 399.65 | 404.05 | 425.16 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-03 10:15:00 | 435.80 | 402.92 | 421.77 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 500.50 | 434.64 | 434.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 519.20 | 466.18 | 455.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 484.35 | 485.01 | 470.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-03 15:15:00 | 492.40 | 485.08 | 470.47 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 472.75 | 486.32 | 474.83 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 423.40 | 480.62 | 480.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 418.40 | 480.00 | 480.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 442.00 | 439.87 | 456.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-19 14:15:00 | 434.05 | 439.80 | 456.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 444.80 | 432.26 | 449.10 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-29 09:15:00 | 434.90 | 432.41 | 449.00 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 434.30 | 425.00 | 440.81 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-10 14:15:00 | 446.00 | 425.61 | 440.73 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 451.85 | 429.04 | 428.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 14:15:00 | 453.20 | 430.90 | 429.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 434.85 | 439.25 | 434.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 09:15:00 | 442.15 | 438.39 | 434.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 442.15 | 438.39 | 434.52 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-26 11:15:00 | 432.85 | 438.31 | 434.52 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 410.70 | 435.79 | 435.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 406.80 | 435.03 | 435.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 435.25 | 421.72 | 427.33 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 510.00 | 432.18 | 432.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 516.95 | 433.03 | 432.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 518.15 | 523.98 | 500.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-27 15:15:00 | 531.00 | 523.67 | 500.99 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-17 14:15:00 | 466.50 | 2024-11-01 17:15:00 | 473.00 | EXIT_EMA400 | -6.50 |
| BUY | 2024-12-30 09:15:00 | 518.45 | 2025-01-06 10:15:00 | 496.50 | EXIT_EMA400 | -21.95 |
| BUY | 2025-01-02 12:15:00 | 506.55 | 2025-01-06 10:15:00 | 496.50 | EXIT_EMA400 | -10.05 |
| SELL | 2025-03-27 10:15:00 | 399.65 | 2025-04-03 10:15:00 | 435.80 | EXIT_EMA400 | -36.15 |
| BUY | 2025-06-03 15:15:00 | 492.40 | 2025-06-16 09:15:00 | 472.75 | EXIT_EMA400 | -19.65 |
| SELL | 2025-08-19 14:15:00 | 434.05 | 2025-09-10 14:15:00 | 446.00 | EXIT_EMA400 | -11.95 |
| SELL | 2025-08-29 09:15:00 | 434.90 | 2025-09-10 14:15:00 | 446.00 | EXIT_EMA400 | -11.10 |
| BUY | 2025-11-26 09:15:00 | 442.15 | 2025-11-26 11:15:00 | 432.85 | EXIT_EMA400 | -9.30 |
