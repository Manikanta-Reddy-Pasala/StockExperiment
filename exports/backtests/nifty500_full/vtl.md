# Vardhman Textiles Ltd. (VTL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 612.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 6 |
| ENTRY1 | 9 |
| ENTRY2 | 2 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 1 / 10
- **Target hits / EMA400 exits:** 1 / 10
- **Total realized P&L (per unit):** -66.32
- **Avg P&L per closed trade:** -6.03

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-27 10:15:00 | 362.30 | 372.22 | 372.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 10:15:00 | 355.80 | 371.52 | 371.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-31 15:15:00 | 370.90 | 370.24 | 371.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-01 09:15:00 | 358.35 | 370.12 | 371.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 367.65 | 365.45 | 368.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-09 14:15:00 | 369.25 | 365.49 | 368.32 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 12:15:00 | 415.30 | 370.47 | 370.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 14:15:00 | 421.70 | 371.42 | 370.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 392.00 | 392.16 | 384.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-12 14:15:00 | 393.95 | 392.17 | 384.12 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-12-20 15:15:00 | 385.00 | 393.91 | 386.62 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 10:15:00 | 488.90 | 498.52 | 498.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 11:15:00 | 487.50 | 498.41 | 498.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 14:15:00 | 483.90 | 476.36 | 484.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 14:15:00 | 466.50 | 476.11 | 483.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 471.70 | 460.44 | 473.16 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-05 14:15:00 | 476.30 | 461.58 | 472.32 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 507.85 | 474.23 | 474.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 518.20 | 477.24 | 475.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 09:15:00 | 509.80 | 510.82 | 496.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-30 09:15:00 | 518.45 | 510.90 | 497.47 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 499.20 | 510.20 | 498.51 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-02 12:15:00 | 506.55 | 510.09 | 498.57 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-01-06 10:15:00 | 496.50 | 510.27 | 499.34 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 13:15:00 | 459.00 | 492.70 | 492.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 14:15:00 | 456.50 | 492.34 | 492.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 10:15:00 | 404.20 | 404.19 | 426.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-27 10:15:00 | 399.65 | 404.13 | 425.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-03 10:15:00 | 435.45 | 402.96 | 421.97 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 13:15:00 | 501.10 | 435.34 | 435.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 519.65 | 466.21 | 455.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 484.35 | 485.02 | 470.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-03 15:15:00 | 490.95 | 485.07 | 470.53 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-16 09:15:00 | 472.75 | 486.31 | 474.87 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 423.40 | 480.61 | 480.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 418.35 | 479.99 | 480.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 442.00 | 440.00 | 456.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-19 14:15:00 | 434.45 | 439.94 | 456.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 444.85 | 432.34 | 449.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-29 09:15:00 | 434.90 | 432.49 | 449.08 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 434.90 | 425.04 | 440.86 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-10 14:15:00 | 446.00 | 425.64 | 440.78 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 451.85 | 429.05 | 429.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 14:15:00 | 453.20 | 430.88 | 429.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 434.85 | 439.26 | 434.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 09:15:00 | 442.50 | 438.41 | 434.54 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 442.50 | 438.41 | 434.54 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-26 11:15:00 | 432.85 | 438.32 | 434.54 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 410.70 | 435.83 | 435.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 406.80 | 435.07 | 435.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 435.20 | 421.75 | 427.37 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 10:15:00 | 503.70 | 432.25 | 432.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 12:15:00 | 523.00 | 453.89 | 444.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 518.10 | 523.96 | 500.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-01 11:15:00 | 529.10 | 523.44 | 501.89 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-01 09:15:00 | 358.35 | 2023-11-09 14:15:00 | 369.25 | EXIT_EMA400 | -10.90 |
| BUY | 2023-12-12 14:15:00 | 393.95 | 2023-12-20 15:15:00 | 385.00 | EXIT_EMA400 | -8.95 |
| SELL | 2024-10-17 14:15:00 | 466.50 | 2024-11-05 14:15:00 | 476.30 | EXIT_EMA400 | -9.80 |
| BUY | 2024-12-30 09:15:00 | 518.45 | 2025-01-06 10:15:00 | 496.50 | EXIT_EMA400 | -21.95 |
| BUY | 2025-01-02 12:15:00 | 506.55 | 2025-01-06 10:15:00 | 496.50 | EXIT_EMA400 | -10.05 |
| SELL | 2025-03-27 10:15:00 | 399.65 | 2025-04-03 10:15:00 | 435.45 | EXIT_EMA400 | -35.80 |
| BUY | 2025-06-03 15:15:00 | 490.95 | 2025-06-16 09:15:00 | 472.75 | EXIT_EMA400 | -18.20 |
| SELL | 2025-08-19 14:15:00 | 434.45 | 2025-09-10 14:15:00 | 446.00 | EXIT_EMA400 | -11.55 |
| SELL | 2025-08-29 09:15:00 | 434.90 | 2025-09-10 14:15:00 | 446.00 | EXIT_EMA400 | -11.10 |
| BUY | 2025-11-26 09:15:00 | 442.50 | 2025-11-26 11:15:00 | 432.85 | EXIT_EMA400 | -9.65 |
| BUY | 2026-04-01 11:15:00 | 529.10 | 2026-04-28 09:15:00 | 610.73 | TARGET | 81.63 |
